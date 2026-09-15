#!/usr/bin/env python3
"""
Standalone precompute of TEST-USER behaviors with local Unsloth Gemma.

This script does ONE thing only:

    train history
        -> behavior windows
        -> local Gemma behavior extraction
        -> save behaviors

It does NOT:
- load a tree
- load state embeddings
- map behaviors to states
- query a reverse tree
- rank candidates
- use negatives or test targets

The output is designed to be reusable by later inference/ranking pipelines.

Per-user output
---------------
{
  "user_id": "...",
  "train_history_item_ids": [...],
  "behavior_windows": [...],
  "generated_behaviors": [
    {
      "task_id": "..._W0",
      "window_index": 0,
      "behavior_explanation": "...",
      "pattern_description": "...",
      "keywords": [...],
      "behavior_signature": "...",
      "mechanism": "...",
      "scope": "...",
      "direction": "...",
      "confidence": 0.85
    },
    ...
  ],
  "recent_behavior_evidence": [
    {
      "behavior_signature": "...",
      "mechanism": "...",
      "scope": "...",
      "direction": "...",
      "confidence": 0.85
    },
    ...
  ]
}

`generated_behaviors` is the full sequence needed later for:
    behavior -> frozen-state mapping -> tree query.

`recent_behavior_evidence` is a compact local-behavior block that can be
inserted directly into a ranking prompt if desired.

Example
-------
python precompute_test_behaviors_gemma.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --window-size 3 \
  --max-train-interactions 30 \
  --recent-behaviors 5 \
  --max-users 300 \
  --output precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --summary-output precomputed/CDs/test_user_behaviors_gemma.summary.json \
  --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm


# =============================================================================
# Generic I/O
# =============================================================================

def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_label(value: Any) -> str:
    x = str(value or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", x).strip()


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


def processed_users(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    out: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("precompute_ok") is True:
                out.add(str(row["user_id"]))

    return out


# =============================================================================
# Item lookup
# =============================================================================

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


def load_items(path: str) -> Dict[Any, Dict[str, Any]]:
    raw = json_load(path)

    try:
        return {
            int(k): v
            for k, v in raw.items()
        }
    except Exception:
        return raw


# =============================================================================
# Behavior windows
# SAME rule as current inference/training pipeline
# =============================================================================

@dataclass
class BehaviorTask:
    task_id: str
    window_index: int
    interaction_sequence: List[Dict[str, str]]


def build_behavior_windows(
    train_item_ids: Sequence[Any],
    items_meta: Dict[Any, Dict[str, Any]],
    window_size: int,
    max_train_interactions: int,
) -> List[List[Dict[str, str]]]:
    train_items = list(train_item_ids)[
        -int(max_train_interactions):
    ]

    history: List[Dict[str, str]] = []
    windows: List[List[Dict[str, str]]] = []

    if not train_items:
        return windows

    for idx, item_id in enumerate(train_items):
        k = resolve_item_key(
            item_id,
            items_meta,
        )

        if k is not None:
            info = items_meta[k]
            history.append({
                "item_id": str(item_id),
                "item": str(
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
                "action": "purchase",
            })

        current_len = len(history)
        should_create = False

        if current_len < window_size:
            if (
                idx == len(train_items) - 1
                and current_len > 0
            ):
                should_create = True
        else:
            if (
                current_len % window_size == 0
                or idx == len(train_items) - 1
            ):
                should_create = True

        if should_create and current_len > 0:
            actual_window = min(
                int(window_size),
                current_len,
            )
            window = [
                dict(x)
                for x in history[-actual_window:]
            ]

            if (
                not windows
                or window != windows[-1]
            ):
                windows.append(window)

    return windows


# =============================================================================
# Behavior prompt
# =============================================================================

BEHAVIOR_SYSTEM = """You are a behavioral memory modeling system for recommender systems.

You produce TWO complementary representations from each short interaction window:

A) Recommendation memory:
   Preserve useful domain-specific preference evidence for downstream recommendation.

B) Cross-user behavior abstraction:
   Describe the behavioral MECHANISM and RELATIONAL SCOPE in a domain-agnostic way,
   so the behavior can later be mapped to canonical states and collaborative
   trajectories.

Never invent evidence not supported by the supplied interactions.
"""


def build_single_behavior_prompt(
    task: BehaviorTask,
) -> str:
    return f"""Analyze this single interaction window.

TASK_ID:
{task.task_id}

INTERACTION_SEQUENCE:
{json.dumps(
    task.interaction_sequence,
    ensure_ascii=False,
    indent=2,
)}

Return ONE JSON object with exactly these fields:
{{
  "task_id": "{task.task_id}",
  "behavior_explanation": "2-3 concise sentences grounded in the observed items",
  "pattern_description": "1-2 concise sentences describing the temporal/relational pattern",
  "keywords": ["3-8", "short", "keywords"],
  "behavior_signature": "short domain-agnostic phrase describing HOW preference evolves",
  "mechanism": "one allowed mechanism label",
  "scope": "one allowed scope label",
  "direction": "one concise direction label",
  "confidence": 0.0
}}

Allowed mechanism labels:
[
  "repetition",
  "persistence",
  "collection expansion",
  "deepening",
  "narrowing",
  "broadening",
  "shifting",
  "returning",
  "adjacent exploration",
  "cross-category exploration",
  "refinement",
  "unknown"
]

Allowed scope labels:
[
  "same item",
  "same creator",
  "same collection/series",
  "same subcategory",
  "same broad category",
  "related category",
  "cross-category",
  "mixed",
  "unknown"
]

Allowed direction examples:
[
  "stable",
  "repeat",
  "deepen",
  "narrow",
  "broaden",
  "shift",
  "return",
  "mixed",
  "unknown"
]

Rules:
1. behavior_explanation and keywords may retain concrete preference evidence.
2. pattern_description should emphasize the temporal/relational pattern.
3. behavior_signature/mechanism/scope/direction must be domain-agnostic.
4. behavior_signature should describe HOW behavior evolves, not WHAT content is preferred.
5. Prefer a specific supported mechanism/scope; use unknown only if evidence is genuinely weak.
6. Output JSON only.
7. Do NOT wrap the object in a list.
8. Do NOT use markdown fences.
"""


# =============================================================================
# Robust JSON parser
# =============================================================================

def extract_json_value(text: str) -> Any:
    s = str(text or "").strip()

    s = re.sub(
        r"^```(?:json)?\s*",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"\s*```$",
        "",
        s,
    ).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = [
        (s.find("{"), "{", "}"),
        (s.find("["), "[", "]"),
    ]
    starts = [
        x for x in starts
        if x[0] >= 0
    ]

    if not starts:
        raise ValueError(
            "No JSON value found"
        )

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

    raise ValueError(
        "Unbalanced JSON"
    )


def normalize_behavior(
    obj: Any,
    task: BehaviorTask,
) -> Dict[str, Any]:
    # Be tolerant if the model still wraps a single object in a list.
    if isinstance(obj, list):
        rows = [
            x
            for x in obj
            if isinstance(x, dict)
        ]
        if not rows:
            raise ValueError(
                f"No behavior object for {task.task_id}"
            )
        obj = rows[0]

    if not isinstance(obj, dict):
        raise ValueError(
            f"Behavior output is not an object "
            f"for {task.task_id}"
        )

    row = dict(obj)

    signature = str(
        row.get("behavior_signature")
        or ""
    ).strip()

    if not signature:
        raise ValueError(
            f"Missing behavior_signature "
            f"for {task.task_id}"
        )

    kws = row.get("keywords", [])
    if not isinstance(kws, list):
        kws = [str(kws)] if kws else []

    try:
        confidence = float(
            row.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0

    return {
        "task_id": task.task_id,
        "window_index": int(
            task.window_index
        ),
        "behavior_explanation": str(
            row.get(
                "behavior_explanation",
                "",
            )
        ).strip(),
        "pattern_description": str(
            row.get(
                "pattern_description",
                "",
            )
        ).strip(),
        "keywords": [
            str(x).strip()
            for x in kws
            if str(x).strip()
        ][:8],
        "behavior_signature": signature,
        "mechanism": normalize_label(
            row.get("mechanism")
            or "unknown"
        ),
        "scope": normalize_label(
            row.get("scope")
            or "unknown"
        ),
        "direction": normalize_label(
            row.get("direction")
            or "unknown"
        ),
        "confidence": max(
            0.0,
            min(1.0, confidence),
        ),
    }


# =============================================================================
# Local Unsloth Gemma
# =============================================================================

@dataclass
class CallStat:
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool
    attempts: int


class LocalGemmaBehaviorExtractor:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        import torch
        from unsloth import FastModel

        self.torch = torch
        self.model_name = model_name
        self.max_seq_length = int(
            max_seq_length
        )
        self.temperature = float(
            temperature
        )
        self.max_new_tokens = int(
            max_new_tokens
        )
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        print(
            f"[INFO] Loading local Gemma: "
            f"{model_name}"
        )

        (
            self.model,
            self.tokenizer,
        ) = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.model.eval()

        if hasattr(
            self.tokenizer,
            "padding_side",
        ):
            self.tokenizer.padding_side = (
                "left"
            )

        if getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        ) is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    BEHAVIOR_SYSTEM
                    + "\n\n"
                    + prompt
                ),
            }],
        }]

        return (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _generate_once(
        self,
        task: BehaviorTask,
        retry_hint: bool,
    ) -> Dict[str, Any]:
        prompt = build_single_behavior_prompt(
            task
        )

        if retry_hint:
            prompt += """
IMPORTANT RETRY:
The previous response could not be parsed.
Return exactly ONE complete JSON object.
Do not output any commentary before or after the JSON.
"""

        rendered = self._render(prompt)

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
            "max_new_tokens": (
                self.max_new_tokens
            ),
            "use_cache": True,
            "pad_token_id": (
                self.tokenizer.pad_token_id
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

        t0 = time.time()

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **kwargs,
            )

        generated = output[:, input_len:]

        raw = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        obj = extract_json_value(raw)
        row = normalize_behavior(
            obj,
            task,
        )

        self.calls.append(
            CallStat(
                input_tokens=input_len,
                output_tokens=int(
                    generated.shape[1]
                ),
                elapsed_sec=float(
                    time.time() - t0
                ),
                parse_ok=True,
                attempts=1,
            )
        )

        return row

    def generate(
        self,
        task: BehaviorTask,
        max_attempts: int,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            max(1, int(max_attempts)) + 1,
        ):
            try:
                return self._generate_once(
                    task,
                    retry_hint=(attempt > 1),
                )
            except Exception as e:
                last_error = e

                print(
                    f"[WARN] {task.task_id} "
                    f"attempt {attempt}/"
                    f"{max_attempts} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        raise ValueError(
            f"Behavior generation failed "
            f"for {task.task_id}: "
            f"{last_error}"
        )

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "provider": (
                    "local_unsloth"
                ),
                "model": self.model_name,
                "calls": 0,
            }

        return {
            "provider": "local_unsloth",
            "model": self.model_name,
            "calls": len(self.calls),
            "input_tokens": int(
                sum(
                    x.input_tokens
                    for x in self.calls
                )
            ),
            "output_tokens": int(
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
        }


# =============================================================================
# Summary
# =============================================================================

def summarize_output(
    output_path: str | Path,
    extractor_stats: Dict[str, Any],
    failed_this_run: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    success_rows: List[
        Dict[str, Any]
    ] = []

    p = Path(output_path)
    if p.exists():
        with p.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if (
                    row.get("precompute_ok")
                    is True
                ):
                    success_rows.append(row)

    mechanism_counts = Counter()
    scope_counts = Counter()
    direction_counts = Counter()
    confidences: List[float] = []
    num_behaviors: List[int] = []

    for row in success_rows:
        behaviors = row.get(
            "generated_behaviors",
            [],
        )

        num_behaviors.append(
            len(behaviors)
        )

        for b in behaviors:
            mechanism_counts[
                str(b.get("mechanism"))
            ] += 1
            scope_counts[
                str(b.get("scope"))
            ] += 1
            direction_counts[
                str(b.get("direction"))
            ] += 1

            if (
                b.get("confidence")
                is not None
            ):
                confidences.append(
                    float(
                        b["confidence"]
                    )
                )

    return {
        "schema_version": (
            "amem_test_behavior_"
            "precompute_summary_v1"
        ),
        "num_users_success": len(
            success_rows
        ),
        "num_behaviors": int(
            sum(num_behaviors)
        ),
        "mean_behaviors_per_user": (
            float(
                np.mean(num_behaviors)
            )
            if num_behaviors
            else 0.0
        ),
        "mean_confidence": (
            float(
                np.mean(confidences)
            )
            if confidences
            else None
        ),
        "top_mechanisms": [
            [k, int(v)]
            for k, v
            in mechanism_counts.most_common()
        ],
        "top_scopes": [
            [k, int(v)]
            for k, v
            in scope_counts.most_common()
        ],
        "top_directions": [
            [k, int(v)]
            for k, v
            in direction_counts.most_common()
        ],
        "failed_this_run": int(
            failed_this_run
        ),
        "local_gemma": (
            extractor_stats
        ),
        "config": config,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone precompute of "
            "test-user behaviors with "
            "local Unsloth Gemma."
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
        "--model",
        default=(
            "unsloth/gemma-3-4b-it-"
            "unsloth-bnb-4bit"
        ),
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
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
        "--max-new-tokens",
        type=int,
        default=512,
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=2,
    )

    p.add_argument(
        "--window-size",
        type=int,
        default=3,
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
        help=(
            "How many latest behaviors to "
            "also store in the compact "
            "ranking evidence block."
        ),
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="0 = all selected users",
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
        "--failures-output",
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

    random.seed(args.seed)
    np.random.seed(args.seed)

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

    if requested_users is None:
        users = list(
            sequences.keys()
        )
    else:
        users = [
            u
            for u in requested_users
            if u in sequences
        ]

    if args.max_users > 0:
        users = users[
            :args.max_users
        ]

    output = Path(args.output)
    summary_output = Path(
        args.summary_output
        or (str(output) + ".summary.json")
    )
    failures_output = Path(
        args.failures_output
        or (str(output) + ".failures.jsonl")
    )

    if (
        output.exists()
        and not args.resume
    ):
        raise RuntimeError(
            f"{output} already exists. "
            "Delete it or use --resume."
        )

    done = (
        processed_users(output)
        if args.resume
        else set()
    )

    pending = [
        uid
        for uid in users
        if uid not in done
    ]

    extractor = (
        LocalGemmaBehaviorExtractor(
            model_name=args.model,
            max_seq_length=(
                args.max_seq_length
            ),
            load_in_4bit=(
                args.load_in_4bit
            ),
            temperature=(
                args.temperature
            ),
            max_new_tokens=(
                args.max_new_tokens
            ),
            seed=args.seed,
        )
    )

    config = {
        "items": args.items,
        "sequences": args.sequences,
        "model": args.model,
        "window_size": (
            args.window_size
        ),
        "max_train_interactions": (
            args.max_train_interactions
        ),
        "recent_behaviors": (
            args.recent_behaviors
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "max_attempts": (
            args.max_attempts
        ),
        "temperature": (
            args.temperature
        ),
        "seed": args.seed,
    }

    print("=" * 90)
    print(
        "STANDALONE TEST-USER "
        "BEHAVIOR PRECOMPUTE"
    )
    print("=" * 90)
    print(
        f"selected users     : "
        f"{len(users)}"
    )
    print(
        f"already completed : "
        f"{len(users) - len(pending)}"
    )
    print(
        f"pending           : "
        f"{len(pending)}"
    )
    print(
        f"window size       : "
        f"{args.window_size}"
    )
    print(
        f"max train history : "
        f"{args.max_train_interactions}"
    )
    print(
        f"output            : "
        f"{output}"
    )

    failed_this_run = 0

    pbar = tqdm(
        pending,
        desc="Precompute test behaviors",
        unit="user",
        dynamic_ncols=True,
    )

    for uid in pbar:
        try:
            user_data = sequences[uid]

            train_ids = list(
                user_data.get(
                    "train",
                    [],
                )
            )

            used_train_ids = train_ids[
                -args.max_train_interactions:
            ]

            windows = (
                build_behavior_windows(
                    train_item_ids=train_ids,
                    items_meta=items_meta,
                    window_size=(
                        args.window_size
                    ),
                    max_train_interactions=(
                        args.max_train_interactions
                    ),
                )
            )

            tasks = [
                BehaviorTask(
                    task_id=f"{uid}_W{i}",
                    window_index=i,
                    interaction_sequence=w,
                )
                for i, w in enumerate(
                    windows
                )
            ]

            behaviors = [
                extractor.generate(
                    task,
                    max_attempts=(
                        args.max_attempts
                    ),
                )
                for task in tasks
            ]

            recent = []

            for b in behaviors[
                -args.recent_behaviors:
            ]:
                recent.append({
                    "window_index": (
                        b["window_index"]
                    ),
                    "behavior_signature": (
                        b[
                            "behavior_signature"
                        ]
                    ),
                    "mechanism": (
                        b["mechanism"]
                    ),
                    "scope": b["scope"],
                    "direction": (
                        b["direction"]
                    ),
                    "confidence": (
                        b["confidence"]
                    ),
                })

            row = {
                "schema_version": (
                    "amem_test_behavior_"
                    "precompute_v1"
                ),
                "precompute_ok": True,
                "user_id": uid,
                "train_history_item_ids": [
                    str(x)
                    for x in used_train_ids
                ],
                "train_history_hash": (
                    stable_hash([
                        str(x)
                        for x in used_train_ids
                    ])
                ),
                "window_size": (
                    args.window_size
                ),
                "max_train_interactions": (
                    args.max_train_interactions
                ),
                "behavior_windows": (
                    windows
                ),
                "generated_behaviors": (
                    behaviors
                ),
                "recent_behavior_evidence": (
                    recent
                ),
            }

            append_jsonl(
                output,
                row,
            )

            pbar.set_postfix(
                behaviors=len(behaviors),
                refresh=False,
            )

        except Exception as e:
            failed_this_run += 1

            failure = {
                "schema_version": (
                    "amem_test_behavior_"
                    "precompute_v1"
                ),
                "precompute_ok": False,
                "user_id": uid,
                "error_type": (
                    type(e).__name__
                ),
                "error": str(e),
            }

            append_jsonl(
                failures_output,
                failure,
            )

            print(
                f"\n[WARN] user={uid} "
                f"failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    pbar.close()

    summary = summarize_output(
        output_path=output,
        extractor_stats=(
            extractor.stats()
        ),
        failed_this_run=(
            failed_this_run
        ),
        config=config,
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
        f"\nOutput   : {output}"
    )
    print(
        f"Failures : "
        f"{failures_output}"
    )
    print(
        f"Summary  : "
        f"{summary_output}"
    )


if __name__ == "__main__":
    main()
