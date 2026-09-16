#!/usr/bin/env python3
"""
Standalone precompute of TEST-USER behaviors with local Unsloth Gemma.

This script does ONE thing only:

    train history
        -> behavior windows
        -> batched local Gemma behavior extraction
        -> save behaviors

Batching is implemented as MULTIPLE INDEPENDENT PROMPTS in one model.generate()
call. Each window still has its own prompt and its own JSON object output.
Windows are NOT combined into one prompt or one JSON array.

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
      "recommendation_behavior": "...",
      "semantic_focus": [...],
      "trajectory_signature": "...",
      "behavior_signature": "...",
      "mechanism": "...",
      "scope": "...",
      "direction": "...",
      "confidence": 0.85,
      "pattern_description": "...",
      "keywords": [...]
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
  --batch-size 8 \
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

SCHEMA_VERSION = "amem_test_behavior_precompute_dual_v2"
BEHAVIOR_PROMPT_VERSION = "dual_view_discriminative_behavior_v1"


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

For each short OBSERVED interaction window, produce TWO complementary views:

A) Recommendation behavior:
   Preserve concrete, discriminative semantic preference evidence that can help
   distinguish relevant candidate items from unrelated items.

B) Trajectory abstraction:
   Describe HOW preference evolves in a domain-agnostic form so behaviors can be
   clustered across users and used in collaborative trajectory prediction.

Never predict a future item.
Never invent evidence not supported by the observed interactions.
Return valid JSON only.
"""


def build_single_behavior_prompt(
    task: BehaviorTask,
) -> str:
    return f"""Analyze this single OBSERVED interaction window.

TASK_ID:
{task.task_id}

INTERACTION_SEQUENCE:
{json.dumps(
    task.interaction_sequence,
    ensure_ascii=False,
    indent=2,
)}

Produce TWO COMPLEMENTARY representations.

A) RECOMMENDATION BEHAVIOR — semantic and discriminative
Describe the concrete preference represented by this observed window.
Preserve useful evidence when supported, such as:
- genre/subgenre or product subcategory;
- style, mood, theme, or use case;
- soundtrack / film / game orientation;
- regional or cultural direction;
- live / compilation / collection characteristics;
- creator continuity when actually visible;
- other concrete semantic properties that can distinguish relevant future items
  from clearly unrelated items.

This view should be specific enough to help a downstream recommender rank candidate
items, while remaining grounded ONLY in this observed window.

B) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative
Describe HOW preference evolves inside this observed window, independently of
the concrete content. This view is used for cluster/state mapping and Tree learning.

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

Allowed direction labels:
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

STRICT RULES:
1. Use ONLY evidence in INTERACTION_SEQUENCE.
2. Do NOT predict or infer a future item.
3. recommendation_behavior MUST preserve concrete semantic evidence when available.
4. Avoid generic phrases such as "exploring related products", "expanding interests",
   "showing varied preferences", or "continuing preferences" when a more specific
   semantic description is supported.
5. semantic_focus should contain short reusable semantic phrases, not full sentences.
6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.
7. Do NOT put exact item/product titles into trajectory_signature.
8. mechanism/scope/direction must remain domain-agnostic.
9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.
10. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.
11. Return exactly ONE JSON object.
12. Do NOT use markdown fences.

Return exactly:
{{
  "task_id": "{task.task_id}",
  "behavior_explanation": "1-2 concise sentences grounded in the observed interactions",
  "recommendation_behavior": "one concise, concrete, semantically discriminative description of the observed preference",
  "semantic_focus": ["3-8", "short", "semantic", "phrases"],
  "trajectory_signature": "short domain-agnostic phrase describing HOW preference evolves",
  "mechanism": "one allowed mechanism label",
  "scope": "one allowed scope label",
  "direction": "one allowed direction label",
  "confidence": 0.0
}}
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
    # Be tolerant if the model wraps a single object in a list.
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

    recommendation_behavior = str(
        row.get("recommendation_behavior")
        or ""
    ).strip()
    if not recommendation_behavior:
        raise ValueError(
            f"Missing recommendation_behavior "
            f"for {task.task_id}"
        )

    semantic_focus = row.get(
        "semantic_focus",
        [],
    )
    if not isinstance(
        semantic_focus,
        list,
    ):
        semantic_focus = (
            [str(semantic_focus)]
            if semantic_focus
            else []
        )

    semantic_focus = [
        str(x).strip()
        for x in semantic_focus
        if str(x).strip()
    ][:8]

    trajectory_signature = str(
        row.get("trajectory_signature")
        or ""
    ).strip()
    if not trajectory_signature:
        raise ValueError(
            f"Missing trajectory_signature "
            f"for {task.task_id}"
        )

    try:
        confidence = float(
            row.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0

    mechanism = normalize_label(
        row.get("mechanism")
        or "unknown"
    )
    scope = normalize_label(
        row.get("scope")
        or "unknown"
    )
    direction = normalize_label(
        row.get("direction")
        or "unknown"
    )

    # -----------------------------------------------------------------
    # Backward compatibility:
    # - pattern_description is the ranking-oriented semantic view.
    # - keywords is the semantic_focus list.
    # - behavior_signature is the Tree-oriented trajectory_signature.
    # -----------------------------------------------------------------
    pattern_description = (
        recommendation_behavior
    )
    keywords = list(
        semantic_focus
    )
    behavior_signature = (
        trajectory_signature
    )

    return {
        "task_id": task.task_id,
        "window_index": int(
            task.window_index
        ),
        "behavior_prompt_version": (
            BEHAVIOR_PROMPT_VERSION
        ),

        "behavior_explanation": str(
            row.get(
                "behavior_explanation",
                "",
            )
        ).strip(),

        # View A: content-aware recommendation signal.
        "recommendation_behavior": (
            recommendation_behavior
        ),
        "semantic_focus": (
            semantic_focus
        ),

        # View B: domain-agnostic trajectory signal.
        "trajectory_signature": (
            trajectory_signature
        ),
        "behavior_signature": (
            behavior_signature
        ),
        "mechanism": mechanism,
        "scope": scope,
        "direction": direction,
        "confidence": max(
            0.0,
            min(1.0, confidence),
        ),

        # Compatibility aliases used by older downstream code.
        "pattern_description": (
            pattern_description
        ),
        "keywords": keywords,
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
        self.max_seq_length = int(max_seq_length)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(
            f"[INFO] Loading local Gemma: "
            f"{model_name}"
        )

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.model.eval()

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": BEHAVIOR_SYSTEM + "\n\n" + prompt,
            }],
        }]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        return kwargs

    def _generate_single_once(
        self,
        task: BehaviorTask,
        retry_hint: bool,
    ) -> Dict[str, Any]:
        prompt = build_single_behavior_prompt(task)

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

        input_len = int(inputs["input_ids"].shape[1])
        kwargs = self._generation_kwargs()

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
        row = normalize_behavior(obj, task)

        self.calls.append(
            CallStat(
                input_tokens=input_len,
                output_tokens=int(generated.shape[1]),
                elapsed_sec=float(time.time() - t0),
                parse_ok=True,
                attempts=1,
            )
        )

        return row

    def generate_single(
        self,
        task: BehaviorTask,
        max_attempts: int,
    ) -> Dict[str, Any]:
        """Robust fallback for one failed sample."""
        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            max(1, int(max_attempts)) + 1,
        ):
            try:
                return self._generate_single_once(
                    task,
                    retry_hint=(attempt > 1),
                )
            except Exception as e:
                last_error = e
                print(
                    f"[WARN] {task.task_id} "
                    f"attempt {attempt}/{max_attempts} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        raise ValueError(
            f"Behavior generation failed "
            f"for {task.task_id}: {last_error}"
        )

    def _generate_batch_once(
        self,
        tasks: List[BehaviorTask],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate multiple independent prompts in one forward generation call.

        Each task has its own prompt and output sequence. No JSON array is used.
        Parse failures are returned as None and retried individually later.
        """
        if not tasks:
            return []

        rendered = [
            self._render(
                build_single_behavior_prompt(task)
            )
            for task in tasks
        ]

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        # With left padding, all generated tokens begin after the common padded
        # input width.
        input_width = int(inputs["input_ids"].shape[1])
        kwargs = self._generation_kwargs()

        t0 = time.time()

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **kwargs,
            )

        generated = outputs[:, input_width:]

        raws = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        elapsed = float(time.time() - t0)
        batch_size = len(tasks)

        results: List[Optional[Dict[str, Any]]] = []

        for task, raw, gen_row in zip(
            tasks,
            raws,
            generated,
        ):
            parse_ok = False
            try:
                obj = extract_json_value(raw.strip())
                row = normalize_behavior(obj, task)
                parse_ok = True
                results.append(row)
            except Exception as e:
                print(
                    f"[WARN] batch parse failed for {task.task_id}: "
                    f"{type(e).__name__}: {e}; retrying individually",
                    flush=True,
                )
                results.append(None)

            # Token count here uses padded input width. It is sufficient for
            # throughput diagnostics and avoids expensive per-sample recounting.
            self.calls.append(
                CallStat(
                    input_tokens=input_width,
                    output_tokens=int(gen_row.shape[0]),
                    elapsed_sec=elapsed / max(1, batch_size),
                    parse_ok=parse_ok,
                    attempts=1,
                )
            )

        return results

    def generate_batch(
        self,
        tasks: List[BehaviorTask],
        batch_size: int,
        max_attempts: int,
    ) -> List[Dict[str, Any]]:
        """
        Batched independent-prompt generation.

        Normal path:
            B independent prompts -> 1 model.generate() -> B independent JSONs.

        If one output fails parsing, only that task is retried individually.
        """
        if not tasks:
            return []

        batch_size = max(1, int(batch_size))
        final_rows: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

        for start in range(0, len(tasks), batch_size):
            end = min(
                start + batch_size,
                len(tasks),
            )
            batch_tasks = tasks[start:end]

            try:
                batch_rows = self._generate_batch_once(
                    batch_tasks
                )
            except Exception as e:
                print(
                    f"[WARN] model.generate batch "
                    f"{start}:{end} failed: "
                    f"{type(e).__name__}: {e}; "
                    "retrying this batch sample-by-sample",
                    flush=True,
                )
                batch_rows = [None] * len(batch_tasks)

            for local_idx, (task, row) in enumerate(
                zip(batch_tasks, batch_rows)
            ):
                global_idx = start + local_idx

                if row is not None:
                    final_rows[global_idx] = row
                    continue

                # Only failed samples pay the slower single-sample retry cost.
                final_rows[global_idx] = self.generate_single(
                    task,
                    max_attempts=max_attempts,
                )

        if any(row is None for row in final_rows):
            raise RuntimeError(
                "Internal error: unresolved behavior row after retries"
            )

        return [
            row
            for row in final_rows
            if row is not None
        ]

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "provider": "local_unsloth",
                "model": self.model_name,
                "calls": 0,
            }

        return {
            "provider": "local_unsloth",
            "model": self.model_name,
            "generation_records": len(self.calls),
            "successful_generation_records": int(
                sum(1 for x in self.calls if x.parse_ok)
            ),
            "parse_failures_before_retry": int(
                sum(1 for x in self.calls if not x.parse_ok)
            ),
            "input_tokens_approx": int(
                sum(x.input_tokens for x in self.calls)
            ),
            "output_tokens": int(
                sum(x.output_tokens for x in self.calls)
            ),
            "elapsed_sec": float(
                sum(x.elapsed_sec for x in self.calls)
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
    semantic_focus_counts = Counter()
    recommendation_behavior_lengths: List[int] = []
    trajectory_signature_lengths: List[int] = []
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

            for sf in b.get(
                "semantic_focus",
                [],
            ):
                semantic_focus_counts[
                    str(sf)
                ] += 1

            recommendation_behavior_lengths.append(
                len(
                    str(
                        b.get(
                            "recommendation_behavior",
                            "",
                        )
                    )
                )
            )
            trajectory_signature_lengths.append(
                len(
                    str(
                        b.get(
                            "trajectory_signature",
                            "",
                        )
                    )
                )
            )

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
            "amem_test_behavior_precompute_dual_summary_v2"
        ),
        "behavior_prompt_version": (
            BEHAVIOR_PROMPT_VERSION
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
        "top_semantic_focus": [
            [k, int(v)]
            for k, v
            in semantic_focus_counts.most_common(30)
        ],
        "mean_recommendation_behavior_chars": (
            float(
                np.mean(
                    recommendation_behavior_lengths
                )
            )
            if recommendation_behavior_lengths
            else 0.0
        ),
        "mean_trajectory_signature_chars": (
            float(
                np.mean(
                    trajectory_signature_lengths
                )
            )
            if trajectory_signature_lengths
            else 0.0
        ),
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
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of independent behavior-window prompts per "
            "model.generate() call. Start with 8; lower if GPU OOM."
        ),
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
        "schema_version": SCHEMA_VERSION,
        "behavior_prompt_version": BEHAVIOR_PROMPT_VERSION,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
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
        "batch_size": (
            args.batch_size
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
        f"batch size        : "
        f"{args.batch_size}"
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

            behaviors = extractor.generate_batch(
                tasks=tasks,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts,
            )

            recent = []

            for b in behaviors[
                -args.recent_behaviors:
            ]:
                recent.append({
                    "window_index": (
                        b["window_index"]
                    ),

                    # Ranking-oriented semantic evidence.
                    "recommendation_behavior": (
                        b[
                            "recommendation_behavior"
                        ]
                    ),
                    "semantic_focus": (
                        b["semantic_focus"]
                    ),

                    # Tree-oriented trajectory abstraction.
                    "trajectory_signature": (
                        b[
                            "trajectory_signature"
                        ]
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
                    SCHEMA_VERSION
                ),
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
                ),
                "behavior_schema": (
                    "dual_view_semantic_plus_trajectory"
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
                    SCHEMA_VERSION
                ),
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
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
