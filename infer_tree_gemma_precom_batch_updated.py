#!/usr/bin/env python3
"""
Standalone Tree-AMEM inference from PRECOMPUTED TEST-USER BEHAVIORS.

This file is fully standalone:
- NO dependency on another inference script
- NO import from infer_tree_amem_gemma_hybrid_v2.py
- NO behavior generation

Pipeline
--------
precomputed test behavior
    -> frozen-state mapping
    -> reverse-tree query
    -> build ranking evidence
    -> local Gemma candidate ranking

Supported state induction modes
-------------------------------
1) hybrid (main method)
   - exact (mechanism, scope) gate when available
   - Qwen cosine retrieval
   - direct top-1 if similarity/margin is clear
   - otherwise Gemma chooses among top-k frozen states
   - no UNK; verifier failure falls back to cosine top-1

2) cluster (KMeans ablation)
   - Qwen embeds the SAME precomputed Gemma behavior text
   - nearest frozen KMeans centroid
   - no verifier
   - tree is queried exactly the same way afterward

Expected behavior cache
-----------------------
Output of:
    precompute_test_behaviors_gemma_batch.py

Example: MAIN HYBRID TREE
-------------------------
python infer_tree_amem_precomputed_behaviors_standalone.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --tree-dir behavior_tree_out_gemini_hybrid \
  --state-mode hybrid \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --llm-batch-size 8 \
  --top-next 3 \
  --max-users 10 \
  --run-baseline \
  --output results/tree_hybrid_precomputed_test10.jsonl \
  --summary-output results/tree_hybrid_precomputed_test10_summary.json

Example: KMEANS-50 TREE ABLATION
--------------------------------
python infer_tree_amem_precomputed_behaviors_standalone.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --tree-dir behavior_tree_out_cluster_k50 \
  --state-mode cluster \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --llm-batch-size 8 \
  --top-next 3 \
  --max-users 10 \
  --run-baseline \
  --output results/tree_cluster50_precomputed_test10.jsonl \
  --summary-output results/tree_cluster50_precomputed_test10_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm


# =============================================================================
# Generic helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(
    d: Dict[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def as_str_list(xs: Any) -> List[str]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        xs = [xs]
    return [str(x) for x in xs]


def stable_user_seed(base_seed: int, user_id: str) -> int:
    h = hashlib.sha1(str(user_id).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(h, 16)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_structured_label(value: Any) -> str:
    x = re.sub(r"\s+", " ", str(value or "").strip().lower())
    x = x.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", x).strip()


def state_key(mechanism: Any, scope: Any) -> str:
    m = normalize_structured_label(mechanism) or "unknown"
    s = normalize_structured_label(scope) or "unknown"
    return f"{m}||{s}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


# =============================================================================
# Data loading / item lookup
# =============================================================================

def load_amem_data(
    items_path: str,
    sequences_path: str,
    negatives_path: str,
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    items_meta = json_load(items_path)
    user_sequences_raw = json_load(sequences_path)
    user_negatives_raw = json_load(negatives_path)

    try:
        items_meta = {int(k): v for k, v in items_meta.items()}
    except (ValueError, TypeError):
        pass

    user_sequences = {
        str(k): v for k, v in user_sequences_raw.items()
    }
    user_negatives = {
        str(k): v for k, v in user_negatives_raw.items()
    }
    return items_meta, user_sequences, user_negatives


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
    k = resolve_item_key(item_id, items_meta)

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
            info.get("title", f"Item {item_id}")
        ),
        "category": str(
            info.get(
                "main_cat",
                info.get("category", "Unknown"),
            )
        ),
    }


# =============================================================================
# Precomputed behavior cache
# =============================================================================

def load_precomputed_behaviors(
    path: str,
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("precompute_ok") is not True:
                continue

            uid = str(row["user_id"])
            behaviors = row.get("generated_behaviors", [])

            if not isinstance(behaviors, list):
                raise ValueError(
                    f"user={uid}: generated_behaviors is not a list"
                )

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
    cached = [
        str(x)
        for x in cache_row.get(
            "train_history_item_ids",
            [],
        )
    ]

    current = [
        str(x)
        for x in list(
            user_data.get("train", [])
        )[-int(max_train_interactions):]
    ]

    if cached == current:
        return

    msg = (
        f"user={uid}: precomputed behavior history does not match "
        f"current train history/max_train_interactions. "
        f"cached_n={len(cached)}, current_n={len(current)}"
    )

    if strict:
        raise ValueError(msg)

    print(f"[WARN] {msg}", flush=True)


# =============================================================================
# JSON extraction for local Gemma outputs
# =============================================================================

def _extract_json_value(text: str) -> Any:
    s = str(text or "").strip()
    s = re.sub(
        r"^```(?:json)?\s*",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = [
        (s.find("{"), "{", "}"),
        (s.find("["), "[", "]"),
    ]
    starts = [x for x in starts if x[0] >= 0]

    if not starts:
        raise ValueError("No JSON value found")

    start, opener, closer = min(
        starts,
        key=lambda x: x[0],
    )

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
                return json.loads(
                    s[start:i + 1]
                )

    raise ValueError("Unbalanced JSON")


# =============================================================================
# Candidate ranking / hybrid-state verifier LLM
# =============================================================================

@dataclass
class LocalCallStat:
    call_type: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool


class LocalGemma:
    """
    This class intentionally has NO behavior-generation method.

    It is used only for:
      1) ambiguous HYBRID state selection among frozen states
      2) final candidate ranking
    """

    def __init__(
        self,
        model: str,
        max_seq_length: int,
        load_in_4bit: bool,
        temperature: float,
        rank_max_new_tokens: int,
        verifier_max_new_tokens: int,
        seed: int,
    ) -> None:
        import torch
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        self.torch = torch
        self.model_name = str(model)
        self.max_seq_length = int(max_seq_length)
        self.temperature = float(temperature)
        self.rank_max_new_tokens = int(rank_max_new_tokens)
        self.verifier_max_new_tokens = int(
            verifier_max_new_tokens
        )
        self.calls: List[LocalCallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(
            f"[INFO] Loading local Gemma: "
            f"{self.model_name}"
        )

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=bool(load_in_4bit),
            load_in_8bit=False,
            full_finetuning=False,
        )

        # Match inference_amem.py exactly for Gemma chat formatting.
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.model.eval()

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        if getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        ) is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

    def _generate_json(
        self,
        system: str,
        prompt: str,
        call_type: str,
        max_new_tokens: int,
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.startswith("<bos>"):
            rendered = rendered[len("<bos>"):]

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        input_len = int(
            inputs["input_ids"].shape[1]
        )

        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
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
                "temperature": self.temperature,
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        t0 = time.time()
        parse_ok = False

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                **kwargs,
            )

        gen = out[:, input_len:]

        raw = self.tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        try:
            obj = _extract_json_value(raw)
            parse_ok = True
            return obj
        finally:
            self.calls.append(
                LocalCallStat(
                    call_type=call_type,
                    input_tokens=input_len,
                    output_tokens=int(
                        gen.shape[1]
                    ),
                    elapsed_sec=float(
                        time.time() - t0
                    ),
                    parse_ok=parse_ok,
                )
            )

    def verify_state(
        self,
        behavior: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], float]:
        allowed = [
            str(x["state_id"])
            for x in candidates
        ]

        if not allowed:
            return None, 0.0

        prompt = f"""Choose the MOST SIMILAR frozen canonical behavior state.

NEW BEHAVIOR:
{json.dumps({
    "behavior_signature": behavior.get("behavior_signature", ""),
    "mechanism": behavior.get("mechanism", ""),
    "scope": behavior.get("scope", ""),
    "direction": behavior.get("direction", ""),
    "pattern_description": behavior.get("pattern_description", ""),
}, ensure_ascii=False, indent=2)}

CANDIDATE FROZEN STATES:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Rules:
- You MUST choose exactly one candidate state_id.
- Compare behavioral evolution/signature first.
- mechanism/scope are useful structural evidence but wording may differ
  between training and test behavior generators.
- Ignore concrete content identity.
- Do not return NEW, UNKNOWN, or any ID outside the supplied candidates.

Allowed state IDs:
{json.dumps(allowed, ensure_ascii=False)}

Return JSON only:
{{"state_id":"one_allowed_id","confidence":0.0}}
"""

        try:
            obj = self._generate_json(
                system=(
                    "You map a behavior to the closest "
                    "frozen canonical behavior state."
                ),
                prompt=prompt,
                call_type="state_verifier",
                max_new_tokens=(
                    self.verifier_max_new_tokens
                ),
            )
        except Exception:
            return None, 0.0

        if not isinstance(obj, dict):
            return None, 0.0

        sid = str(
            obj.get("state_id", "")
        ).strip()

        try:
            conf = max(
                0.0,
                min(
                    1.0,
                    float(
                        obj.get(
                            "confidence",
                            0.0,
                        )
                    ),
                ),
            )
        except Exception:
            conf = 0.0

        if sid not in allowed:
            return None, conf

        return sid, conf

    def _build_rank_prompt(
        self,
        *,
        history_items: List[Dict[str, str]],
        candidate_items: List[Dict[str, str]],
        tree_evidence: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, List[str]]:
        """Build the SAME base ranking prompt used by inference_amem.py.

        NativeLLM receives no collaborative block. Tree-AMEM receives its
        tree evidence in the same secondary-evidence slot used by AMEM.
        """
        ids = [
            str(x["item_id"])
            for x in candidate_items
        ]
        n = len(ids)

        evidence_block = ""
        if tree_evidence is not None:
            evidence_block = (
                "\nCollaborative memory (secondary evidence):\n"
                + json.dumps(
                    tree_evidence,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        prompt = f"""You are a recommendation ranking system.

Rank ALL candidate items for the target user.

Priority:
1. Match the user's recent preferences and behavioral pattern.
2. Use category and semantic compatibility.
3. Use retrieved collaborative memory as additional evidence when relevant.

User recent history (most recent last):
{json.dumps(history_items[-10:], indent=2, ensure_ascii=False)}
{evidence_block}

Candidate items:
{json.dumps(candidate_items, indent=2, ensure_ascii=False)}

Requirements:
- Rank ALL {n} candidates.
- Each candidate item_id must appear exactly once.
- Do not invent item IDs.
- Return ONLY valid JSON.

{{"ranked_item_ids": ["id1", "..."], "reasoning": "1 concise sentence"}}"""

        system = "You are a recommendation ranking system."
        return system, prompt, ids

    def rank(
        self,
        *,
        history_items: List[Dict[str, str]],
        candidate_items: List[Dict[str, str]],
        tree_evidence: Optional[Dict[str, Any]],
        call_type: str,
    ) -> Tuple[List[str], Dict[str, Any]]:
        system, prompt, ids = self._build_rank_prompt(
            history_items=history_items,
            candidate_items=candidate_items,
            tree_evidence=tree_evidence,
        )

        obj = self._generate_json(
            system=system,
            prompt=prompt,
            call_type=call_type,
            max_new_tokens=self.rank_max_new_tokens,
        )

        if not isinstance(obj, dict):
            raise ValueError(
                "Ranking output is not an object"
            )

        ranked = sanitize_ranking(
            obj.get("ranked_item_ids", []),
            ids,
        )

        return ranked, {
            "reasoning": str(
                obj.get("reasoning", "")
            ),
            "parse_ok": True,
            "batched": False,
        }

    def _generate_json_batch(
        self,
        *,
        systems: List[str],
        prompts: List[str],
        call_type: str,
        max_new_tokens: int,
    ) -> List[Optional[Any]]:
        """
        Multiple independent prompts in ONE model.generate() call.

        Each user still has a separate prompt and a separate JSON output.
        Users are NOT combined into one prompt.
        """
        if len(systems) != len(prompts):
            raise ValueError(
                "systems/prompts length mismatch"
            )

        if not prompts:
            return []

        rendered: List[str] = []

        for system, prompt in zip(
            systems,
            prompts,
        ):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if text.startswith("<bos>"):
                text = text[len("<bos>"):]
            rendered.append(text)

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        # padding_side='left', therefore generated tokens start after
        # this common padded input width for every sample.
        input_width = int(
            inputs["input_ids"].shape[1]
        )

        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
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
                "temperature": self.temperature,
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        t0 = time.time()

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                **kwargs,
            )

        generated = out[:, input_width:]

        raws = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        elapsed = float(
            time.time() - t0
        )

        batch_n = len(prompts)
        results: List[Optional[Any]] = []

        for raw, gen_row in zip(
            raws,
            generated,
        ):
            parse_ok = False
            obj: Optional[Any] = None

            try:
                obj = _extract_json_value(
                    raw.strip()
                )
                parse_ok = True
            except Exception:
                obj = None

            self.calls.append(
                LocalCallStat(
                    call_type=call_type,
                    # approximate per-sample padded input width
                    input_tokens=input_width,
                    output_tokens=int(
                        gen_row.shape[0]
                    ),
                    elapsed_sec=(
                        elapsed
                        / max(
                            1,
                            batch_n,
                        )
                    ),
                    parse_ok=parse_ok,
                )
            )

            results.append(obj)

        return results

    def rank_batch(
        self,
        jobs: List[Dict[str, Any]],
        *,
        batch_size: int,
        call_type: str,
    ) -> List[
        Tuple[
            List[str],
            Dict[str, Any],
        ]
    ]:
        """
        Batch ranking across users.

        `jobs` entries:
        {
          "history_items": [...],
          "candidate_items": [...],
          "tree_evidence": {...} or None
        }

        On a batch-level generation failure, that batch falls back to
        individual rank() calls. If only one sample has malformed JSON,
        only that sample is retried individually.
        """
        if not jobs:
            return []

        batch_size = max(
            1,
            int(batch_size),
        )

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

        for start in range(
            0,
            len(jobs),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(jobs),
            )

            chunk = jobs[
                start:end
            ]

            systems: List[str] = []
            prompts: List[str] = []
            candidate_id_lists: List[
                List[str]
            ] = []

            for job in chunk:
                (
                    system,
                    prompt,
                    ids,
                ) = self._build_rank_prompt(
                    history_items=job[
                        "history_items"
                    ],
                    candidate_items=job[
                        "candidate_items"
                    ],
                    tree_evidence=job.get(
                        "tree_evidence"
                    ),
                )

                systems.append(system)
                prompts.append(prompt)
                candidate_id_lists.append(
                    ids
                )

            try:
                objs = self._generate_json_batch(
                    systems=systems,
                    prompts=prompts,
                    call_type=call_type,
                    max_new_tokens=(
                        self.rank_max_new_tokens
                    ),
                )
            except Exception as e:
                print(
                    f"[WARN] ranking batch "
                    f"{start}:{end} failed: "
                    f"{type(e).__name__}: {e}; "
                    "falling back to individual ranking",
                    flush=True,
                )
                objs = [
                    None
                    for _ in chunk
                ]

            for local_idx, (
                job,
                ids,
                obj,
            ) in enumerate(
                zip(
                    chunk,
                    candidate_id_lists,
                    objs,
                )
            ):
                global_idx = (
                    start + local_idx
                )

                if isinstance(
                    obj,
                    dict,
                ):
                    ranked = sanitize_ranking(
                        obj.get(
                            "ranked_item_ids",
                            [],
                        ),
                        ids,
                    )

                    final[
                        global_idx
                    ] = (
                        ranked,
                        {
                            "reasoning": str(
                                obj.get(
                                    "reasoning",
                                    "",
                                )
                            ),
                            "parse_ok": True,
                            "batched": True,
                        },
                    )
                    continue

                # Match inference_amem.py: malformed JSON falls back to the
                # persisted candidate input order instead of making another
                # stochastic/numerically-different generation call.
                print(
                    f"[WARN] ranking parse failed for batch sample "
                    f"index={global_idx}; falling back to candidate order",
                    flush=True,
                )
                final[global_idx] = (
                    list(ids),
                    {
                        "reasoning": "",
                        "parse_ok": False,
                        "batched": True,
                        "fallback": "candidate_order",
                    },
                )

        if any(
            x is None
            for x in final
        ):
            raise RuntimeError(
                "Internal error: unresolved batch ranking output"
            )

        return [
            x
            for x in final
            if x is not None
        ]

    def stats(self) -> Dict[str, Any]:
        by: Dict[str, Dict[str, Any]] = {}

        for c in self.calls:
            x = by.setdefault(
                c.call_type,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_sec": 0.0,
                    "parse_failures": 0,
                },
            )

            x["calls"] += 1
            x["input_tokens"] += c.input_tokens
            x["output_tokens"] += c.output_tokens
            x["elapsed_sec"] += c.elapsed_sec

            if not c.parse_ok:
                x["parse_failures"] += 1

        return {
            "provider": "local_unsloth",
            "model": self.model_name,
            "calls": len(self.calls),
            "input_tokens": int(
                sum(
                    c.input_tokens
                    for c in self.calls
                )
            ),
            "output_tokens": int(
                sum(
                    c.output_tokens
                    for c in self.calls
                )
            ),
            "elapsed_sec": float(
                sum(
                    c.elapsed_sec
                    for c in self.calls
                )
            ),
            "parse_failures": int(
                sum(
                    1
                    for c in self.calls
                    if not c.parse_ok
                )
            ),
            "by_type": by,
        }


# =============================================================================
# Shared embedding encoder
# =============================================================================

class QwenEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        import torch
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
            f"[INFO] Loading state embedding model="
            f"{self.model_name} device={self.device}"
        )

        self.encoder = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, 0),
                dtype=np.float32,
            )

        x = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            x,
            dtype=np.float32,
        )


# =============================================================================
# Main HYBRID state vocabulary
# =============================================================================

class HybridStateVocabulary:
    def __init__(
        self,
        states_json: str,
        state_embeddings_npy: str,
        embedding_model: Optional[str],
        embedding_device: str,
        embedding_batch_size: int,
        low_threshold: Optional[float],
        high_threshold: Optional[float],
        top_k: Optional[int],
        verifier_min_confidence: Optional[float],
    ) -> None:
        payload = json_load(states_json)

        if (
            isinstance(payload, dict)
            and isinstance(
                payload.get("states"),
                list,
            )
        ):
            self.rows = [
                dict(x)
                for x in payload["states"]
            ]
            self.metadata = payload.get(
                "metadata",
                {},
            )
        elif isinstance(payload, list):
            self.rows = [
                dict(x)
                for x in payload
            ]
            self.metadata = {}
        else:
            raise ValueError(
                f"Cannot parse states JSON: "
                f"{states_json}"
            )

        self.state_by_id: Dict[
            str,
            Dict[str, Any],
        ] = {}
        self.pair_to_indices: Dict[
            str,
            List[int],
        ] = defaultdict(list)

        for i, r in enumerate(self.rows):
            sid = str(r["state_id"])

            mechanism = normalize_structured_label(
                r.get("canonical_mechanism")
                or r.get("mechanism")
            )
            scope = normalize_structured_label(
                r.get("canonical_scope")
                or r.get("scope")
            )
            key = state_key(
                mechanism,
                scope,
            )

            r["_mechanism"] = mechanism
            r["_scope"] = scope
            r["_pair_key"] = key
            r["_index"] = i

            self.state_by_id[sid] = r
            self.pair_to_indices[key].append(i)

        self.state_embeddings = np.asarray(
            np.load(state_embeddings_npy),
            dtype=np.float32,
        )

        if len(self.state_embeddings) != len(
            self.rows
        ):
            raise ValueError(
                "State embedding/state row mismatch"
            )

        self.state_embeddings = l2_normalize(
            self.state_embeddings
        )

        self.low_threshold = float(
            low_threshold
            if low_threshold is not None
            else self.metadata.get(
                "state_low_threshold",
                0.70,
            )
        )

        self.high_threshold = float(
            high_threshold
            if high_threshold is not None
            else self.metadata.get(
                "state_high_threshold",
                0.92,
            )
        )

        self.top_k = int(
            top_k
            if top_k is not None
            else self.metadata.get(
                "state_top_k",
                3,
            )
        )

        self.verifier_min_confidence = float(
            verifier_min_confidence
            if verifier_min_confidence
            is not None
            else self.metadata.get(
                "verifier_min_confidence",
                0.70,
            )
        )

        self.embedding_model_name = str(
            embedding_model
            or self.metadata.get(
                "embedding_model"
            )
            or "Qwen/Qwen3-Embedding-0.6B"
        )

        self.encoder = QwenEncoder(
            model_name=(
                self.embedding_model_name
            ),
            device=embedding_device,
            batch_size=embedding_batch_size,
        )

    def state_text(
        self,
        sid: Optional[str],
    ) -> str:
        if (
            not sid
            or sid not in self.state_by_id
        ):
            return "UNKNOWN"

        r = self.state_by_id[sid]

        return (
            f"{r.get('_mechanism', '')} | "
            f"{r.get('_scope', '')} | "
            f"{str(r.get('canonical_behavior_signature', '') or '').strip()}"
        )

    def state_evidence(
        self,
        sid: Optional[str],
    ) -> Dict[str, Any]:
        if (
            not sid
            or sid not in self.state_by_id
        ):
            return {
                "state_id": None,
                "mechanism": "unknown",
                "scope": "unknown",
                "representative_signature": "",
            }

        r = self.state_by_id[sid]

        return {
            "state_id": sid,
            "mechanism": r.get(
                "_mechanism",
                "",
            ),
            "scope": r.get(
                "_scope",
                "",
            ),
            "representative_signature": str(
                r.get(
                    "canonical_behavior_signature",
                    "",
                )
                or ""
            ).strip(),
        }

    def _candidate(
        self,
        idx: int,
        sim: float,
    ) -> Dict[str, Any]:
        r = self.rows[idx]

        return {
            "state_id": str(
                r["state_id"]
            ),
            "similarity": round(
                float(sim),
                4,
            ),
            "representative_behavior_signature": str(
                r.get(
                    "canonical_behavior_signature",
                    "",
                )
                or ""
            ).strip(),
            "representative_direction": str(
                r.get(
                    "canonical_direction",
                    "",
                )
                or ""
            ).strip(),
            "representative_pattern_description": str(
                r.get(
                    "representative_pattern_description",
                    "",
                )
                or ""
            ).strip(),
        }

    def map_behaviors(
        self,
        behaviors: List[Dict[str, Any]],
        llm: LocalGemma,
        margin_threshold: float,
    ) -> List[Dict[str, Any]]:
        sigs = [
            str(
                b.get(
                    "behavior_signature",
                    "",
                )
                or ""
            ).strip()
            for b in behaviors
        ]

        if any(not x for x in sigs):
            raise ValueError(
                "Hybrid state mapping requires "
                "behavior_signature for every behavior"
            )

        emb = self.encoder.encode(sigs)
        out: List[Dict[str, Any]] = []
        all_indices = list(
            range(len(self.rows))
        )

        for i, behavior in enumerate(
            behaviors
        ):
            mechanism = (
                normalize_structured_label(
                    behavior.get(
                        "mechanism"
                    )
                )
            )
            scope = (
                normalize_structured_label(
                    behavior.get("scope")
                )
            )

            pair_key = state_key(
                mechanism,
                scope,
            )

            pair_indices = (
                self.pair_to_indices.get(
                    pair_key,
                    [],
                )
            )

            if pair_indices:
                candidate_indices = (
                    pair_indices
                )
                retrieval_scope = (
                    "exact_pair"
                )
            else:
                candidate_indices = (
                    all_indices
                )
                retrieval_scope = (
                    "global_fallback"
                )

            scores = (
                self.state_embeddings[
                    candidate_indices
                ]
                @ emb[i]
            )

            order = np.argsort(-scores)

            k = min(
                max(1, self.top_k),
                len(candidate_indices),
            )

            ranked = [
                (
                    candidate_indices[
                        int(j)
                    ],
                    float(
                        scores[int(j)]
                    ),
                )
                for j in order[:k]
            ]

            best_idx, best_sim = (
                ranked[0]
            )

            second_sim = (
                ranked[1][1]
                if len(ranked) > 1
                else -1.0
            )

            margin = float(
                best_sim - second_sim
            )

            base: Dict[str, Any] = {
                "state_id": None,
                "pair_key": pair_key,
                "state_key": pair_key,
                "mechanism": mechanism,
                "scope": scope,
                "direction": (
                    normalize_structured_label(
                        behavior.get(
                            "direction"
                        )
                    )
                ),
                "behavior_signature": sigs[i],
                "confidence": float(
                    behavior.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                ),
                "retrieval_scope": (
                    retrieval_scope
                ),
                "best_similarity": float(
                    best_sim
                ),
                "second_similarity": (
                    float(second_sim)
                    if len(ranked) > 1
                    else None
                ),
                "similarity_margin": (
                    margin
                ),
                "verifier_confidence": None,
            }

            if (
                best_sim
                >= self.high_threshold
                or margin
                >= float(
                    margin_threshold
                )
                or len(ranked) == 1
            ):
                sid = str(
                    self.rows[
                        best_idx
                    ]["state_id"]
                )

                base.update({
                    "state_id": sid,
                    "mapping_reason": (
                        "high_similarity_direct_map"
                        if best_sim
                        >= self.high_threshold
                        else (
                            "clear_margin_direct_map"
                        )
                    ),
                    "state_text": (
                        self.state_text(
                            sid
                        )
                    ),
                })

                out.append(base)
                continue

            candidates = [
                self._candidate(
                    idx,
                    sim,
                )
                for idx, sim
                in ranked
            ]

            sid, verifier_conf = (
                llm.verify_state(
                    behavior,
                    candidates,
                )
            )

            base[
                "verifier_confidence"
            ] = float(verifier_conf)

            if sid is None:
                sid = str(
                    self.rows[
                        best_idx
                    ]["state_id"]
                )
                reason = (
                    "verifier_failed_"
                    "cosine_top1_fallback"
                )
            else:
                reason = (
                    "gemma_selected_"
                    "frozen_state"
                )

            base.update({
                "state_id": sid,
                "mapping_reason": reason,
                "state_text": (
                    self.state_text(sid)
                ),
            })

            out.append(base)

        return out


# =============================================================================
# KMeans cluster-state vocabulary
# =============================================================================

class ClusterStateVocabulary:
    def __init__(
        self,
        states_json: str,
        state_embeddings_npy: str,
        embedding_model: Optional[str],
        embedding_device: str,
        embedding_batch_size: int,
        cluster_text_field: Optional[str],
    ) -> None:
        payload = json_load(states_json)

        if not (
            isinstance(payload, dict)
            and isinstance(
                payload.get("states"),
                list,
            )
        ):
            raise ValueError(
                "Cluster states JSON must contain "
                "{metadata, states}"
            )

        self.rows = [
            dict(x)
            for x in payload["states"]
        ]
        self.metadata = payload.get(
            "metadata",
            {},
        )

        self.state_by_id = {
            str(r["state_id"]): r
            for r in self.rows
        }

        self.state_embeddings = (
            np.asarray(
                np.load(
                    state_embeddings_npy
                ),
                dtype=np.float32,
            )
        )

        if len(self.state_embeddings) != len(
            self.rows
        ):
            raise ValueError(
                "Cluster centroid/state row mismatch"
            )

        self.state_embeddings = l2_normalize(
            self.state_embeddings
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
            or "pattern_description"
        )

        self.encoder = QwenEncoder(
            model_name=(
                self.embedding_model_name
            ),
            device=embedding_device,
            batch_size=embedding_batch_size,
        )

    def _behavior_text(
        self,
        b: Dict[str, Any],
    ) -> str:
        if (
            self.cluster_text_field
            == "behavior_signature"
        ):
            text = str(
                b.get(
                    "behavior_signature",
                    "",
                )
                or ""
            ).strip()

        elif (
            self.cluster_text_field
            == "combined"
        ):
            pattern = str(
                b.get(
                    "pattern_description",
                    "",
                )
                or ""
            ).strip()

            explanation = str(
                b.get(
                    "behavior_explanation",
                    "",
                )
                or ""
            ).strip()

            text = (
                f"Behavior pattern: {pattern}. "
                f"Evidence summary: {explanation}"
            ).strip()

        else:
            text = str(
                b.get(
                    "pattern_description",
                    "",
                )
                or ""
            ).strip()

        if not text:
            # Safe fallback only when the requested raw field is absent.
            text = str(
                b.get(
                    "behavior_signature",
                    "",
                )
                or ""
            ).strip()

        if not text:
            raise ValueError(
                "Cluster mapping needs non-empty "
                "behavior text"
            )

        return text

    def state_evidence(
        self,
        sid: Optional[str],
    ) -> Dict[str, Any]:
        if (
            not sid
            or sid not in self.state_by_id
        ):
            return {
                "state_id": None,
                "mechanism": "unknown",
                "scope": "unknown",
                "representative_signature": "",
            }

        r = self.state_by_id[sid]

        return {
            "state_id": sid,
            "mechanism": "cluster",
            "scope": "cluster",
            "representative_signature": str(
                r.get(
                    "canonical_behavior_signature",
                    r.get(
                        "canonical_text",
                        "",
                    ),
                )
                or ""
            ).strip(),
        }

    def map_behaviors(
        self,
        behaviors: List[Dict[str, Any]],
        llm: LocalGemma,
        margin_threshold: float,
    ) -> List[Dict[str, Any]]:
        del llm
        del margin_threshold

        texts = [
            self._behavior_text(b)
            for b in behaviors
        ]

        emb = self.encoder.encode(
            texts
        )

        scores = (
            emb
            @ self.state_embeddings.T
        )

        out: List[Dict[str, Any]] = []

        for i, behavior in enumerate(
            behaviors
        ):
            order = np.argsort(
                -scores[i]
            )

            best_idx = int(order[0])
            best_sim = float(
                scores[i, best_idx]
            )

            second_sim = (
                float(
                    scores[
                        i,
                        int(order[1]),
                    ]
                )
                if len(order) > 1
                else None
            )

            sid = str(
                self.rows[
                    best_idx
                ]["state_id"]
            )

            out.append({
                "state_id": sid,
                "mechanism": (
                    normalize_structured_label(
                        behavior.get(
                            "mechanism"
                        )
                    )
                ),
                "scope": (
                    normalize_structured_label(
                        behavior.get(
                            "scope"
                        )
                    )
                ),
                "direction": (
                    normalize_structured_label(
                        behavior.get(
                            "direction"
                        )
                    )
                ),
                "behavior_signature": str(
                    behavior.get(
                        "behavior_signature",
                        "",
                    )
                    or ""
                ).strip(),
                "cluster_text": texts[i],
                "confidence": float(
                    behavior.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                ),
                "retrieval_scope": (
                    "global_cluster_centroids"
                ),
                "best_similarity": (
                    best_sim
                ),
                "second_similarity": (
                    second_sim
                ),
                "similarity_margin": (
                    best_sim
                    - second_sim
                    if second_sim
                    is not None
                    else None
                ),
                "verifier_confidence": None,
                "mapping_reason": (
                    "nearest_kmeans_centroid"
                ),
                "state_text": str(
                    self.state_by_id[
                        sid
                    ].get(
                        "canonical_text",
                        sid,
                    )
                ),
            })

        return out


# =============================================================================
# Reverse suffix tree query
# =============================================================================

class ReverseBehaviorTree:
    def __init__(
        self,
        tree_json: str,
        vocab: Any,
    ) -> None:
        payload = json_load(
            tree_json
        )

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
                f"Tree root "
                f"{self.root_id!r} missing"
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

        traversed_reverse: List[
            str
        ] = []

        stop_reason = (
            "history_exhausted"
        )

        for token in reversed(recent):
            children = (
                self.nodes[cur].get(
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
            deepest_active = self.root_id

        structural_node = (
            self.nodes[
                deepest_structural
            ]
        )

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

        preds: List[
            Dict[str, Any]
        ] = []

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
                self.vocab.state_evidence(
                    sid
                )
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

        matched_context = [
            self.vocab.state_evidence(
                str(s)
            )
            for s in active_node.get(
                "context_states",
                [],
            )
        ]

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
            "deepest_structural_context": (
                structural_node.get(
                    "context_states",
                    [],
                )
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
            "matched_context": (
                matched_context
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
            "next_behaviors": preds,
        }


def build_tree_evidence(
    mappings: List[Dict[str, Any]],
    tree_result: Dict[str, Any],
    recent_count: int,
) -> Dict[str, Any]:
    recent = []

    for m in mappings[
        -int(recent_count):
    ]:
        recent.append({
            "state_id": (
                m.get("state_id")
            ),
            "mechanism": (
                m.get("mechanism")
            ),
            "scope": (
                m.get("scope")
            ),
            "signature": (
                m.get(
                    "behavior_signature"
                )
            ),
        })

    return {
        "recent_behavior_states": (
            recent
        ),
        "matched_behavior_context": {
            "order": (
                tree_result[
                    "matched_order"
                ]
            ),
            "states": (
                tree_result[
                    "matched_context"
                ]
            ),
            "support_users": (
                tree_result[
                    "matched_support_users"
                ]
            ),
            "support_occurrences": (
                tree_result[
                    "matched_support_occurrences"
                ]
            ),
        },
        "predicted_next_behaviors": (
            tree_result[
                "next_behaviors"
            ]
        ),
    }


# =============================================================================
# Candidate construction
# =============================================================================

def get_candidates_for_user(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_file_rows: Optional[
        Dict[str, Dict[str, Any]]
    ],
    seed: int,
) -> Tuple[List[str], List[str]]:
    if (
        candidate_file_rows
        is not None
        and str(user_id)
        in candidate_file_rows
    ):
        row = candidate_file_rows[
            str(user_id)
        ]

        candidates = as_str_list(
            row.get(
                "candidates",
                row.get(
                    "candidate_item_ids",
                    [],
                ),
            )
        )

        target = as_str_list(
            row.get(
                "target",
                row.get(
                    "ground_truth",
                    row.get(
                        "ground_truth_item_ids",
                        [],
                    ),
                ),
            )
        )

        if not target:
            target = as_str_list(
                user_data.get(
                    "test",
                    [],
                )
            )

        if not candidates:
            raise ValueError(
                f"Candidate file has empty "
                f"candidates for user {user_id}"
            )

        return candidates, target

    ground_truth = as_str_list(
        user_data.get(
            "test",
            [],
        )
    )

    negatives = as_str_list(
        negative_data.get(
            "test_neg",
            [],
        )
    )

    candidates = (
        ground_truth + negatives
    )

    # Canonical shuffle shared with inference_amem.py.
    rng = random.Random(f"{seed}:{user_id}")
    rng.shuffle(candidates)

    return candidates, ground_truth


def load_candidate_file(
    path: Optional[str],
) -> Optional[
    Dict[str, Dict[str, Any]]
]:
    if not path:
        return None

    obj = json_load(path)
    rows: Dict[
        str,
        Dict[str, Any],
    ] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                rows[str(k)] = v

    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(
                row,
                dict,
            ):
                continue

            uid = first_present(
                row,
                [
                    "user_id",
                    "uid",
                    "user",
                ],
            )

            if uid is not None:
                rows[str(uid)] = row

    if not rows:
        raise ValueError(
            "Could not parse candidate file"
        )

    return rows


def resolve_candidate_file_path(
    candidate_file: Optional[str],
    sequences_path: str,
    seed: int,
) -> str:
    """Resolve one shared candidate cache path for Tree and NativeLLM runs."""
    if candidate_file:
        return str(candidate_file)

    seq = Path(sequences_path)
    return str(
        seq.with_name(
            f"{seq.stem}_candidates_seed{int(seed)}.json"
        )
    )


def ensure_candidate_cache(
    path: str,
    users: List[str],
    user_sequences: Dict[str, Any],
    user_negatives: Dict[str, Any],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Create missing candidate rows once, persist them, then always reuse them.

    The canonical candidate order is exactly the one used by inference_amem.py:
    test item(s) + test negatives, shuffled with Random(f"{seed}:{user_id}").
    Existing rows are never reshuffled.
    """
    p = Path(path)
    rows: Dict[str, Dict[str, Any]] = {}

    if p.exists():
        loaded = load_candidate_file(str(p))
        rows = loaded or {}

    changed = False
    for uid in users:
        suid = str(uid)
        user_data = user_sequences[suid]
        negative_data = user_negatives.get(suid, {})
        target = as_str_list(user_data.get("test", []))
        negatives = as_str_list(negative_data.get("test_neg", []))
        if not target:
            raise ValueError(f"user={suid}: no test item for candidate cache")

        expected = target + negatives
        if suid in rows:
            cached = as_str_list(
                rows[suid].get(
                    "candidates",
                    rows[suid].get("candidate_item_ids", []),
                )
            )
            cached_target = as_str_list(
                rows[suid].get(
                    "target",
                    rows[suid].get(
                        "ground_truth",
                        rows[suid].get("ground_truth_item_ids", []),
                    ),
                )
            )
            if not cached or sorted(cached) != sorted(expected):
                raise ValueError(
                    f"user={suid}: existing candidate cache does not match "
                    "current test + test_neg. Delete/rebuild the cache or use "
                    "a different --candidate-file."
                )
            if cached_target and cached_target != target:
                raise ValueError(
                    f"user={suid}: cached target differs from current test target"
                )
            continue

        candidates = list(expected)
        rng = random.Random(f"{seed}:{suid}")
        rng.shuffle(candidates)

        rows[suid] = {
            "user_id": suid,
            "candidates": candidates,
            "target": target,
        }
        changed = True

    if changed or not p.exists():
        json_dump(p, rows)
        print(
            f"[INFO] Candidate cache saved: {p} "
            f"({len(rows)} users)",
            flush=True,
        )
    else:
        print(
            f"[INFO] Candidate cache loaded: {p} "
            f"({len(rows)} users)",
            flush=True,
        )

    missing = [str(u) for u in users if str(u) not in rows]
    if missing:
        raise RuntimeError(
            f"Candidate cache missing {len(missing)} selected users; "
            f"examples={missing[:10]}"
        )

    return rows


def sanitize_ranking(
    raw_ids: Iterable[Any],
    candidate_ids: List[str],
) -> List[str]:
    valid = set(candidate_ids)
    seen = set()
    out: List[str] = []

    for x in raw_ids:
        s = str(x)

        if (
            s in valid
            and s not in seen
        ):
            out.append(s)
            seen.add(s)

    for s in candidate_ids:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out


# =============================================================================
# Metrics
# =============================================================================

def recall_at_k(
    pred: List[str],
    gt: List[str],
    k: int,
) -> float:
    if not gt:
        return 0.0

    return (
        len(
            set(pred[:k])
            & set(gt)
        )
        / len(set(gt))
    )


def ndcg_at_k(
    pred: List[str],
    gt: List[str],
    k: int,
) -> float:
    gt_set = set(gt)

    if not gt_set:
        return 0.0

    dcg = 0.0

    for i, x in enumerate(
        pred[:k]
    ):
        if x in gt_set:
            dcg += (
                1.0
                / math.log2(i + 2)
            )

    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(
            min(
                len(gt_set),
                k,
            )
        )
    )

    return (
        dcg / idcg
        if idcg > 0
        else 0.0
    )


def target_rank(
    pred: List[str],
    gt: List[str],
) -> Optional[int]:
    gt_set = set(gt)

    for i, x in enumerate(
        pred,
        start=1,
    ):
        if x in gt_set:
            return i

    return None


def ranking_metrics(
    pred: List[str],
    gt: List[str],
) -> Dict[str, Any]:
    return {
        "recall@5": recall_at_k(
            pred,
            gt,
            5,
        ),
        "recall@10": recall_at_k(
            pred,
            gt,
            10,
        ),
        "recall@20": recall_at_k(
            pred,
            gt,
            20,
        ),
        "ndcg@5": ndcg_at_k(
            pred,
            gt,
            5,
        ),
        "ndcg@10": ndcg_at_k(
            pred,
            gt,
            10,
        ),
        "ndcg@20": ndcg_at_k(
            pred,
            gt,
            20,
        ),
        "target_rank": target_rank(
            pred,
            gt,
        ),
    }


def aggregate_metrics(
    rows: List[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, float]]:
    vals = [
        r.get(key)
        for r in rows
        if isinstance(
            r.get(key),
            dict,
        )
    ]

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
        xs = [
            float(v[metric])
            for v in vals
            if v.get(metric)
            is not None
        ]

        out[metric] = (
            float(np.mean(xs))
            if xs
            else 0.0
        )

    ranks = [
        float(
            v["target_rank"]
        )
        for v in vals
        if v.get(
            "target_rank"
        )
        is not None
    ]

    out["mean_target_rank"] = (
        float(np.mean(ranks))
        if ranks
        else float("nan")
    )

    return out


# =============================================================================
# User selection / resume
# =============================================================================

def load_user_ids_file(
    path: Optional[str],
) -> Optional[List[str]]:
    if not path:
        return None

    text = Path(path).read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    try:
        obj = json.loads(text)

        if isinstance(obj, list):
            return [
                str(x)
                for x in obj
            ]

        if isinstance(obj, dict):
            if isinstance(
                obj.get("users"),
                list,
            ):
                return [
                    str(x)
                    for x in obj["users"]
                ]

            return [
                str(k)
                for k in obj.keys()
            ]

    except json.JSONDecodeError:
        pass

    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


def processed_users_from_jsonl(
    path: str,
) -> set[str]:
    p = Path(path)

    if not p.exists():
        return set()

    out: set[str] = set()

    with p.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                if (
                    row.get("user_id")
                    is not None
                ):
                    out.add(
                        str(
                            row[
                                "user_id"
                            ]
                        )
                    )
            except Exception:
                continue

    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone Tree-AMEM inference from "
            "precomputed test-user behaviors."
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
        help=(
            "Shared candidate cache. If omitted, a deterministic cache is created "
            "next to --sequences and reused on later runs."
        ),
    )

    p.add_argument(
        "--precomputed-behaviors",
        default=None,
        help=(
            "JSONL output of precompute_test_behaviors_gemma_batch.py. "
            "Required for tree modes; not needed with --baseline-only."
        ),
    )

    p.add_argument(
        "--tree-dir",
        default=None,
        help=(
            "Tree directory. Required for tree modes; "
            "not needed with --baseline-only."
        ),
    )
    p.add_argument(
        "--state-mode",
        choices=[
            "hybrid",
            "cluster",
        ],
        default="hybrid",
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
    )
    p.add_argument(
        "--embedding-device",
        default="auto",
    )
    p.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
    )

    # Hybrid-state mapping options.
    p.add_argument(
        "--state-low-threshold",
        type=float,
        default=None,
    )
    p.add_argument(
        "--state-high-threshold",
        type=float,
        default=None,
    )
    p.add_argument(
        "--state-top-k",
        type=int,
        default=None,
    )
    p.add_argument(
        "--verifier-min-confidence",
        type=float,
        default=None,
    )
    p.add_argument(
        "--state-margin-threshold",
        type=float,
        default=0.05,
    )

    # Cluster-mode option. If omitted, read from behavior_states.json metadata.
    p.add_argument(
        "--cluster-text-field",
        choices=[
            "pattern_description",
            "behavior_signature",
            "combined",
        ],
        default=None,
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
        action=(
            argparse.BooleanOptionalAction
        ),
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
        "--llm-batch-size",
        type=int,
        default=1,
        help=(
            "Number of independent user-ranking prompts per model.generate() call. "
            "Default=1 to match inference_amem.py exactly; increase only for speed tests."
        ),
    )
    p.add_argument(
        "--verifier-max-new-tokens",
        type=int,
        default=96,
    )

    p.add_argument(
        "--max-train-interactions",
        type=int,
        default=30,
    )
    p.add_argument(
        "--recent-behaviors",
        type=int,
        default=5,
    )
    p.add_argument(
        "--top-next",
        type=int,
        default=3,
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
    )

    p.add_argument(
        "--run-baseline",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=False,
        help=(
            "Run Tree-AMEM and ALSO run the NativeLLM baseline."
        ),
    )
    p.add_argument(
        "--baseline-only",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=False,
        help=(
            "Run ONLY NativeLLM: History + Candidates -> LLM. "
            "Do not load tree, Qwen, state vocabulary, or memory evidence."
        ),
    )

    p.add_argument(
        "--strict-cache-history",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=True,
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
        action=(
            argparse.BooleanOptionalAction
        ),
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

    if args.baseline_only and args.run_baseline:
        print(
            "[INFO] --baseline-only is enabled; "
            "--run-baseline is redundant and will be ignored."
        )

    states_json = None
    state_embeddings = None
    tree_json = None

    if not args.baseline_only:
        if not args.precomputed_behaviors:
            raise ValueError(
                "--precomputed-behaviors is required unless --baseline-only is used"
            )

        if not args.tree_dir:
            raise ValueError(
                "--tree-dir is required unless --baseline-only is used"
            )

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

    required_files = [
        args.items,
        args.sequences,
        args.negatives,
    ]

    if not args.baseline_only:
        required_files.extend([
            args.precomputed_behaviors,
            states_json,
            state_embeddings,
            tree_json,
        ])

    for required in required_files:
        if required is None or not Path(required).exists():
            raise FileNotFoundError(
                required
            )

    behavior_cache: Dict[str, Dict[str, Any]] = {}

    if not args.baseline_only:
        behavior_cache = (
            load_precomputed_behaviors(
                args.precomputed_behaviors
            )
        )

    (
        items_meta,
        user_sequences,
        user_negatives,
    ) = load_amem_data(
        args.items,
        args.sequences,
        args.negatives,
    )

    llm = LocalGemma(
        model=args.model,
        max_seq_length=(
            args.max_seq_length
        ),
        load_in_4bit=(
            args.load_in_4bit
        ),
        temperature=(
            args.temperature
        ),
        rank_max_new_tokens=(
            args.rank_max_new_tokens
        ),
        verifier_max_new_tokens=(
            args.verifier_max_new_tokens
        ),
        seed=args.seed,
    )

    vocab = None
    tree = None

    if not args.baseline_only:
        if args.state_mode == "hybrid":
            vocab = (
                HybridStateVocabulary(
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
                    low_threshold=(
                        args.state_low_threshold
                    ),
                    high_threshold=(
                        args.state_high_threshold
                    ),
                    top_k=(
                        args.state_top_k
                    ),
                    verifier_min_confidence=(
                        args.verifier_min_confidence
                    ),
                )
            )
        else:
            vocab = (
                ClusterStateVocabulary(
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
                )
            )

        tree = ReverseBehaviorTree(
            tree_json=tree_json,
            vocab=vocab,
        )

    requested = load_user_ids_file(
        args.user_ids_file
    )

    # Select users ONLY from sequence order first.  Do not silently skip a
    # missing behavior-cache user and pull a later user into the first N.
    if requested is None:
        users = [str(uid) for uid in user_sequences.keys()]
    else:
        users = [
            str(uid)
            for uid in requested
            if str(uid) in user_sequences
        ]

    if args.max_users > 0:
        users = users[:args.max_users]

    if not users:
        raise ValueError("No selected users found in sequences")

    if not args.baseline_only:
        missing_behaviors = [
            uid for uid in users
            if uid not in behavior_cache
        ]
        if missing_behaviors:
            raise ValueError(
                "Selected users must be identical across Tree and NativeLLM, "
                "but precomputed behaviors are missing for "
                f"{len(missing_behaviors)} selected users; "
                f"examples={missing_behaviors[:10]}. "
                "Precompute these users instead of silently replacing them."
            )

    # Always use a persisted shared candidate cache. If it does not yet exist,
    # create it deterministically for the selected users, save it, then use it.
    candidate_file_path = resolve_candidate_file_path(
        args.candidate_file,
        args.sequences,
        args.seed,
    )
    candidate_rows = ensure_candidate_cache(
        path=candidate_file_path,
        users=users,
        user_sequences=user_sequences,
        user_negatives=user_negatives,
        seed=args.seed,
    )
    args.candidate_file = candidate_file_path

    output_path = Path(
        args.output
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
            args.output
        )
        if args.resume
        else set()
    )

    pending = [
        u
        for u in users
        if u not in done
    ]

    print("=" * 96)
    print(
        "STANDALONE TREE-AMEM INFERENCE "
        "FROM PRECOMPUTED BEHAVIORS"
    )
    print("=" * 96)
    run_mode = (
        "baseline-only"
        if args.baseline_only
        else (
            "tree+baseline"
            if args.run_baseline
            else "tree-only"
        )
    )

    print(
        f"run mode            : "
        f"{run_mode}"
    )

    if not args.baseline_only:
        print(
            f"state mode          : "
            f"{args.state_mode}"
        )
        print(
            f"tree dir            : "
            f"{args.tree_dir}"
        )
        print(
            f"cached behavior users: "
            f"{len(behavior_cache)}"
        )
    print(
        f"selected users      : "
        f"{len(users)}"
    )
    print(
        f"already completed   : "
        f"{len(users) - len(pending)}"
    )
    print(
        f"pending             : "
        f"{len(pending)}"
    )
    print(
        f"top-next            : "
        f"{args.top_next}"
    )
    print(
        f"ranking model       : "
        f"{args.model}"
    )
    print(
        f"LLM batch size      : "
        f"{args.llm_batch_size}"
    )

    failures = 0
    t0 = time.time()

    pbar = tqdm(
        total=len(pending),
        desc=(
            "NativeLLM baseline"
            if args.baseline_only
            else f"{args.state_mode} tree ranking"
        ),
        unit="user",
        dynamic_ncols=True,
    )

    # Process users in chunks so final LLM ranking can be batched.
    for batch_start in range(
        0,
        len(pending),
        max(
            1,
            int(args.llm_batch_size),
        ),
    ):
        batch_uids = pending[
            batch_start:
            batch_start
            + max(
                1,
                int(args.llm_batch_size),
            )
        ]

        prepared: List[
            Dict[str, Any]
        ] = []

        # ----------------------------------------------------------
        # Phase A: non-ranking preparation for each user.
        # Cluster/Qwen mapping is already efficient; hybrid verifier
        # may still make individual calls for ambiguous states.
        # ----------------------------------------------------------
        for uid in batch_uids:
            try:
                user_data = (
                    user_sequences[uid]
                )

                behaviors = None
                mappings = None
                state_sequence = None
                tree_result = None
                tree_evidence = None

                if not args.baseline_only:
                    cache_row = (
                        behavior_cache[uid]
                    )

                    validate_behavior_cache_history(
                        uid=uid,
                        cache_row=cache_row,
                        user_data=user_data,
                        max_train_interactions=(
                            args.max_train_interactions
                        ),
                        strict=(
                            args.strict_cache_history
                        ),
                    )

                    behaviors = [
                        dict(x)
                        for x in cache_row.get(
                            "generated_behaviors",
                            [],
                        )
                    ]

                    if not behaviors:
                        raise ValueError(
                            f"user={uid}: "
                            "no precomputed behaviors"
                        )

                    mappings = (
                        vocab.map_behaviors(
                            behaviors=behaviors,
                            llm=llm,
                            margin_threshold=(
                                args.state_margin_threshold
                            ),
                        )
                    )

                    state_sequence = [
                        str(
                            m["state_id"]
                        )
                        for m in mappings
                    ]

                    tree_result = (
                        tree.query(
                            state_sequence=(
                                state_sequence
                            ),
                            top_next=(
                                args.top_next
                            ),
                        )
                    )

                    tree_evidence = (
                        build_tree_evidence(
                            mappings=mappings,
                            tree_result=(
                                tree_result
                            ),
                            recent_count=(
                                args.recent_behaviors
                            ),
                        )
                    )

                negative_data = (
                    user_negatives.get(
                        uid,
                        {},
                    )
                )

                (
                    candidate_ids,
                    ground_truth,
                ) = get_candidates_for_user(
                    user_id=uid,
                    user_data=user_data,
                    negative_data=(
                        negative_data
                    ),
                    candidate_file_rows=(
                        candidate_rows
                    ),
                    seed=args.seed,
                )

                candidate_items = [
                    get_item_info(
                        i,
                        items_meta,
                    )
                    for i in candidate_ids
                ]

                train_ids = list(
                    user_data.get(
                        "train",
                        [],
                    )
                )

                history_items = [
                    get_item_info(
                        i,
                        items_meta,
                    )
                    for i in train_ids[
                        -args.max_train_interactions:
                    ]
                ]

                prepared.append({
                    "user_id": uid,
                    "behaviors": behaviors,
                    "mappings": mappings,
                    "state_sequence": (
                        state_sequence
                    ),
                    "tree_result": (
                        tree_result
                    ),
                    "tree_evidence": (
                        tree_evidence
                    ),
                    "candidate_ids": (
                        candidate_ids
                    ),
                    "ground_truth": (
                        ground_truth
                    ),
                    "candidate_items": (
                        candidate_items
                    ),
                    "history_items": (
                        history_items
                    ),
                })

            except Exception as e:
                failures += 1

                print(
                    f"\n[WARN] user={uid} "
                    f"failed during preparation: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        # ----------------------------------------------------------
        # Phase B: batched final LLM ranking.
        # ----------------------------------------------------------
        tree_results_batch: List[
            Optional[
                Tuple[
                    List[str],
                    Dict[str, Any],
                ]
            ]
        ] = [
            None
            for _ in prepared
        ]

        baseline_results_batch: List[
            Optional[
                Tuple[
                    List[str],
                    Dict[str, Any],
                ]
            ]
        ] = [
            None
            for _ in prepared
        ]

        if (
            prepared
            and not args.baseline_only
        ):
            tree_jobs = [
                {
                    "history_items": x[
                        "history_items"
                    ],
                    "candidate_items": x[
                        "candidate_items"
                    ],
                    "tree_evidence": x[
                        "tree_evidence"
                    ],
                }
                for x in prepared
            ]

            try:
                ranked = llm.rank_batch(
                    tree_jobs,
                    batch_size=(
                        args.llm_batch_size
                    ),
                    call_type=(
                        "tree_ranking"
                    ),
                )

                tree_results_batch = [
                    x
                    for x in ranked
                ]

            except Exception as e:
                failures += len(
                    prepared
                )

                print(
                    f"\n[WARN] tree ranking "
                    f"batch failed completely: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

                # Mark these prepared users as failed;
                # no rows will be written below.
                tree_results_batch = [
                    None
                    for _ in prepared
                ]

        if (
            prepared
            and (
                args.baseline_only
                or args.run_baseline
            )
        ):
            baseline_jobs = [
                {
                    "history_items": x[
                        "history_items"
                    ],
                    "candidate_items": x[
                        "candidate_items"
                    ],
                    "tree_evidence": None,
                }
                for x in prepared
            ]

            try:
                ranked = llm.rank_batch(
                    baseline_jobs,
                    batch_size=(
                        args.llm_batch_size
                    ),
                    call_type=(
                        "baseline_ranking"
                    ),
                )

                baseline_results_batch = [
                    x
                    for x in ranked
                ]

            except Exception as e:
                failures += len(
                    prepared
                )

                print(
                    f"\n[WARN] baseline ranking "
                    f"batch failed completely: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

                baseline_results_batch = [
                    None
                    for _ in prepared
                ]

        # ----------------------------------------------------------
        # Phase C: metrics + write rows.
        # ----------------------------------------------------------
        for idx, item in enumerate(
            prepared
        ):
            uid = item[
                "user_id"
            ]

            tree_ranked = None
            tree_rank_meta = None
            tree_metrics = None

            baseline_ranked = None
            baseline_meta = None
            baseline_metrics = None

            try:
                if not args.baseline_only:
                    tree_pair = (
                        tree_results_batch[
                            idx
                        ]
                    )

                    if tree_pair is None:
                        raise RuntimeError(
                            "Tree ranking result "
                            "missing for user"
                        )

                    (
                        tree_ranked,
                        tree_rank_meta,
                    ) = tree_pair

                    tree_metrics = (
                        ranking_metrics(
                            tree_ranked,
                            item[
                                "ground_truth"
                            ],
                        )
                    )

                if (
                    args.baseline_only
                    or args.run_baseline
                ):
                    baseline_pair = (
                        baseline_results_batch[
                            idx
                        ]
                    )

                    if baseline_pair is None:
                        raise RuntimeError(
                            "Baseline ranking "
                            "result missing for user"
                        )

                    (
                        baseline_ranked,
                        baseline_meta,
                    ) = baseline_pair

                    baseline_metrics = (
                        ranking_metrics(
                            baseline_ranked,
                            item[
                                "ground_truth"
                            ],
                        )
                    )

                row = {
                    "schema_version": (
                        "amem_precomputed_behavior_"
                        "tree_inference_v2_batch"
                    ),
                    "user_id": uid,
                    "run_mode": (
                        "baseline-only"
                        if args.baseline_only
                        else (
                            "tree+baseline"
                            if args.run_baseline
                            else "tree-only"
                        )
                    ),
                    "state_mode": (
                        None
                        if args.baseline_only
                        else args.state_mode
                    ),
                    "behavior_cache_user_id": (
                        None
                        if args.baseline_only
                        else uid
                    ),
                    "generated_behaviors": (
                        item[
                            "behaviors"
                        ]
                    ),
                    "state_mappings": (
                        item[
                            "mappings"
                        ]
                    ),
                    "state_sequence": (
                        item[
                            "state_sequence"
                        ]
                    ),
                    "tree_result": (
                        item[
                            "tree_result"
                        ]
                    ),
                    "tree_evidence": (
                        item[
                            "tree_evidence"
                        ]
                    ),
                    "candidate_item_ids": (
                        item[
                            "candidate_ids"
                        ]
                    ),
                    "ground_truth_item_ids": (
                        item[
                            "ground_truth"
                        ]
                    ),
                    "tree_ranked_item_ids": (
                        tree_ranked
                    ),
                    "tree_rank_meta": (
                        tree_rank_meta
                    ),
                    "tree_metrics": (
                        tree_metrics
                    ),
                    "baseline_ranked_item_ids": (
                        baseline_ranked
                    ),
                    "baseline_rank_meta": (
                        baseline_meta
                    ),
                    "baseline_metrics": (
                        baseline_metrics
                    ),
                }

                append_jsonl(
                    args.output,
                    row,
                )

                if args.baseline_only:
                    pbar.set_postfix(
                        rank=(
                            baseline_metrics.get(
                                "target_rank"
                            )
                            if baseline_metrics
                            else None
                        ),
                        refresh=False,
                    )
                else:
                    tr = (
                        item[
                            "tree_result"
                        ]
                        or {}
                    )

                    pbar.set_postfix(
                        order=tr.get(
                            "matched_order",
                            0,
                        ),
                        next=len(
                            tr.get(
                                "next_behaviors",
                                [],
                            )
                        ),
                        rank=(
                            tree_metrics.get(
                                "target_rank"
                            )
                            if tree_metrics
                            else None
                        ),
                        refresh=False,
                    )

            except Exception as e:
                failures += 1

                print(
                    f"\n[WARN] user={uid} "
                    f"failed during ranking/write: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        pbar.update(
            len(batch_uids)
        )

    pbar.close()

    # Aggregate the complete output file, including prior resume rows.
    all_rows: List[
        Dict[str, Any]
    ] = []

    if output_path.exists():
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    all_rows.append(
                        json.loads(line)
                    )
                except Exception:
                    pass

    tree_metrics = aggregate_metrics(
        all_rows,
        "tree_metrics",
    )

    baseline_metrics = (
        aggregate_metrics(
            all_rows,
            "baseline_metrics",
        )
    )

    tree_minus_baseline = None

    if (
        tree_metrics is not None
        and baseline_metrics
        is not None
    ):
        tree_minus_baseline = {
            "recall@5": (
                tree_metrics["recall@5"]
                - baseline_metrics["recall@5"]
            ),
            "recall@10": (
                tree_metrics["recall@10"]
                - baseline_metrics["recall@10"]
            ),
            "recall@20": (
                tree_metrics["recall@20"]
                - baseline_metrics["recall@20"]
            ),
            "ndcg@5": (
                tree_metrics["ndcg@5"]
                - baseline_metrics["ndcg@5"]
            ),
            "ndcg@10": (
                tree_metrics["ndcg@10"]
                - baseline_metrics["ndcg@10"]
            ),
            "ndcg@20": (
                tree_metrics["ndcg@20"]
                - baseline_metrics["ndcg@20"]
            ),
            "mean_target_rank_improvement": (
                baseline_metrics[
                    "mean_target_rank"
                ]
                - tree_metrics[
                    "mean_target_rank"
                ]
            ),
        }

    order_counts = Counter()
    mapping_reason_counts = Counter()
    retrieval_scope_counts = Counter()
    root_fallbacks = 0
    total_tree_rows = 0
    total_mappings = 0

    for row in all_rows:
        tr = row.get(
            "tree_result"
        )

        if isinstance(tr, dict):
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

        for m in (
            row.get(
                "state_mappings"
            )
            or []
        ):
            total_mappings += 1

            mapping_reason_counts[
                str(
                    m.get(
                        "mapping_reason",
                        "",
                    )
                )
            ] += 1

            retrieval_scope_counts[
                str(
                    m.get(
                        "retrieval_scope",
                        "",
                    )
                )
            ] += 1

    summary = {
        "schema_version": (
            "amem_precomputed_behavior_"
            "tree_inference_summary_v1"
        ),
        "run_mode": (
            "baseline-only"
            if args.baseline_only
            else (
                "tree+baseline"
                if args.run_baseline
                else "tree-only"
            )
        ),
        "state_mode": (
            None
            if args.baseline_only
            else args.state_mode
        ),
        "num_users": len(
            all_rows
        ),
        "selected_user_ids": users,
        "failed_this_run": (
            failures
        ),
        "tree_metrics": (
            tree_metrics
        ),
        "baseline_metrics": (
            baseline_metrics
        ),
        "tree_minus_baseline": (
            tree_minus_baseline
        ),
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
        "state_mapping_reason_counts": (
            dict(
                mapping_reason_counts
            )
        ),
        "state_retrieval_scope_counts": (
            dict(
                retrieval_scope_counts
            )
        ),
        "total_state_mappings": (
            total_mappings
        ),
        "local_gemma": (
            llm.stats()
        ),
        "elapsed_sec_this_run": float(
            time.time() - t0
        ),
        "config": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "candidate_file": (
                args.candidate_file
            ),
            "precomputed_behaviors": (
                None
                if args.baseline_only
                else args.precomputed_behaviors
            ),
            "tree_dir": (
                None
                if args.baseline_only
                else args.tree_dir
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
            "state_mode": (
                args.state_mode
            ),
            "top_next": (
                args.top_next
            ),
            "recent_behaviors": (
                args.recent_behaviors
            ),
            "max_train_interactions": (
                args.max_train_interactions
            ),
            "model": args.model,
            "llm_batch_size": (
                args.llm_batch_size
            ),
            "run_baseline": (
                args.run_baseline
            ),
            "seed": args.seed,
        },
    }

    summary_output = (
        args.summary_output
        or (
            str(args.output)
            + ".summary.json"
        )
    )

    json_dump(
        summary_output,
        summary,
    )

    print("\nSummary")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nOutput  : "
        f"{args.output}"
    )
    print(
        f"Summary : "
        f"{summary_output}"
    )


if __name__ == "__main__":
    main()
