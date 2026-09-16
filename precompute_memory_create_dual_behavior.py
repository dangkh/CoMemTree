#!/usr/bin/env python3
"""
Precompute the AMem LOCAL MEMORY CREATE stage with batched Unsloth Gemma.

This script intentionally stores ONLY semantic/text memory artifacts:
    train interactions
        -> interaction windows
        -> Gemma dual-view behavior extraction
        -> recommendation semantics + trajectory abstraction
        -> local_memories_gemma.jsonl

It DOES NOT:
    - create embeddings
    - load an embedding model
    - perform cosine search
    - link memories
    - evolve memories
    - build the global memory pool

Embeddings are intentionally deferred to the global-memory build stage, where
all local-memory texts can be encoded in one batch. This keeps the precomputed
artifact compact and allows the embedding model to be changed without rerunning
Gemma extraction.

The window construction follows the original AMem code. For example, with
window_size=3 and 8 valid train interactions, the windows are:
    [1,2,3], [4,5,6], [6,7,8]

Example
-------
python precompute_memory_create_gemma_text.py \
  --data_dir /home/ducnm2/AgenticRec_CFmemory/data/CDs \
  --sequences_file user_sequences_10_5000.json \
  --items_file items.json \
  --output precomputed/CDs/local_memories_gemma_text.jsonl \
  --model_name /home/hkieu/.cache/huggingface/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe \
  --number_of_users 100 \
  --window_size 3 \
  --llm_batch_size 8 \
  --max_new_tokens 256

Suggested packages
------------------
pip install -U unsloth tqdm

Notes
-----
- No Hugging Face token is hard-coded.
- Use `huggingface-cli login` or HF_TOKEN if authentication is needed.
- `--number_of_users <= 0` means all users.
- The output is JSONL: one local memory per line.
- Use a NEW output filename when switching from the older schema that stored
  embeddings. Resume will reject an old embedding-containing JSONL to avoid
  mixing schemas.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Import Unsloth before other Transformers-dependent model imports.
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


SCHEMA_VERSION = "amem_local_memory_dual_behavior_v4"
BEHAVIOR_PROMPT_VERSION = "dual_view_discriminative_behavior_v2_taxonomy"


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
# Prompt and JSON parsing
# =============================================================================

def build_behavior_extraction_prompt(
    interaction_summary: List[Dict[str, Any]]
) -> str:
    """
    Create TWO complementary behavior views from the SAME observed window:

    1) recommendation_behavior / semantic_focus:
       content-aware and discriminative enough to help downstream item ranking.

    2) trajectory_signature + mechanism/scope/direction:
       domain-agnostic abstraction used for cross-user clustering and Tree learning.

    IMPORTANT: this stage only sees OBSERVED interactions. It never sees a future
    test/ground-truth item, so there is no oracle leakage.
    """
    return f"""You are creating a behavioral memory from a SHORT OBSERVED purchase window.

OBSERVED INTERACTIONS:
{json.dumps(interaction_summary, indent=2, ensure_ascii=False)}

Produce TWO COMPLEMENTARY representations.

A) RECOMMENDATION BEHAVIOR — semantic and discriminative
Describe the concrete preference represented by this observed window.
Preserve useful evidence when supported, such as:
- genre/subgenre or product subcategory;
- style, mood, theme, or use case;
- soundtrack / film / game orientation;
- regional or cultural direction;
- live / compilation / collection characteristics;
- creator continuity when it is actually visible;
- other concrete semantic properties that can distinguish relevant future items
  from clearly unrelated items.

This view should be specific enough to help a downstream recommender rank candidate
items, but it must remain grounded ONLY in the observed interactions.

B) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative
Describe HOW the preference evolves within this observed window, independently of
the concrete content. This view is used to cluster behaviors across users and learn
behavior transitions.

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


TAXONOMY — USE THESE DEFINITIONS STRICTLY

MECHANISM:
- repetition:
  Literal repetition of the EXACT SAME item_id inside the observed window.
  NEVER use repetition merely because different items are similar.
- persistence:
  Different item_ids maintain the same broad preference without a clear move
  toward a narrower/more specific preference.
- collection expansion:
  Different item_ids expand an identifiable creator/series/collection/family.
- deepening:
  Different item_ids move further into the same focused subcategory/style/theme.
- narrowing:
  Preference becomes more selective/specific than earlier interactions.
- broadening:
  Preference expands to a wider range while retaining a recognizable anchor.
- shifting:
  A clear move from one preference/category/style toward another.
- returning:
  The user comes back to an earlier preference after an intervening different one.
- adjacent exploration:
  Movement into a clearly related neighboring category/style.
- cross-category exploration:
  Movement across substantially different categories/styles.
- refinement:
  Different item_ids explore variants inside a very narrow preference.
- unknown:
  Only when the observed evidence is genuinely insufficient.

SCOPE:
- same item:
  ONLY when the exact same item_id occurs more than once.
- same creator:
  Different item_ids but clear creator/artist continuity is visible.
- same collection/series:
  Different item_ids within an identifiable series/collection/family.
- same subcategory:
  Different item_ids sharing a focused genre/subgenre/product subtype.
- same broad category:
  Different item_ids sharing only a broad category.
- related category:
  Different but clearly neighboring categories.
- cross-category:
  Clearly different categories.
- mixed:
  No single relational scope dominates.
- unknown:
  Evidence is insufficient.

DIRECTION:
- repeat:
  ONLY for literal repeat of an exact item_id.
- stable:
  Different item_ids maintain a broadly stable preference.
- deepen:
  Move further into a focused preference.
- narrow:
  Move toward a more specific subset.
- broaden:
  Expand the preference range.
- shift:
  Move toward a different preference.
- return:
  Revisit an earlier preference after leaving it.
- mixed:
  Multiple directions coexist.
- unknown:
  Evidence is insufficient.

CRITICAL DISAMBIGUATION:
- Different item_ids that are semantically similar are NOT "repetition".
- Different albums/products from the same creator are NOT "same item".
- Different items in the same focused style should usually be persistence,
  deepening, refinement, or collection expansion depending on the evidence.
- Use "repetition + same item + repeat" only when an exact item_id repeats.

STRICT RULES:
1. Use ONLY evidence present in the observed interactions.
2. Do NOT predict or invent the user's future item.
3. recommendation_behavior MUST preserve concrete semantic evidence when available.
4. Avoid generic phrases such as "exploring related products", "expanding interests",
   "showing varied preferences", or "continuing preferences" when a more specific
   semantic description is supported.
5. semantic_focus should contain short reusable semantic phrases, not full sentences.
6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.
7. Do NOT put artist/item/product titles into trajectory_signature.
8. mechanism/scope/direction must be domain-agnostic.
9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.
10. BEFORE choosing repetition/same item/repeat, explicitly check whether an exact
    item_id occurs more than once. If all item_ids are distinct, those labels are forbidden.
11. When item_ids are distinct, distinguish persistence vs collection expansion vs
    deepening/refinement vs broadening/shifting from the semantic relations in the window.
12. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.
13. Return JSON only. No markdown.

Return exactly:
{{
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



def enforce_structural_consistency(
    mechanism: str,
    scope: str,
    direction: str,
    interaction_sequence: List[Dict[str, Any]],
) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
    """
    Deterministic guardrail for labels with literal structural meaning.

    We do NOT try to infer deepening/collection-expansion automatically.
    We only prevent impossible repetition/same-item/repeat labels when no
    exact item_id repeats. This keeps semantic interpretation with Gemma while
    protecting the Tree from taxonomy collapse.
    """
    item_ids = [
        str(x.get("item_id", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_id", "")).strip()
    ]
    unique_ids = set(item_ids)
    has_exact_repeat = (
        len(item_ids) >= 2
        and len(unique_ids) < len(item_ids)
    )
    all_same_item = (
        len(item_ids) >= 2
        and len(unique_ids) == 1
    )

    categories = [
        str(x.get("item_category", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_category", "")).strip()
        and str(x.get("item_category", "")).strip().lower() != "unknown"
    ]
    same_known_category = (
        len(categories) >= 2
        and len(set(categories)) == 1
    )

    adjustments: List[str] = []

    mechanism = str(mechanism or "unknown").strip().lower()
    scope = str(scope or "unknown").strip().lower()
    direction = str(direction or "unknown").strip().lower()

    if not has_exact_repeat:
        if mechanism == "repetition":
            old = mechanism
            # Do not guess persistence/deepening here. The item category can be
            # too broad (e.g., all CDs), so an impossible literal repetition is
            # converted to unknown and exposed for audit.
            mechanism = "unknown"
            adjustments.append(
                f"mechanism:{old}->{mechanism}:no_exact_item_repeat"
            )

        if scope == "same item":
            old = scope
            scope = "mixed"
            adjustments.append(
                f"scope:{old}->{scope}:no_exact_item_repeat"
            )

        if direction == "repeat":
            old = direction
            direction = "unknown"
            adjustments.append(
                f"direction:{old}->{direction}:no_exact_item_repeat"
            )

    # If every observed interaction is literally the same item, these labels
    # have unambiguous structural support.
    if all_same_item:
        if mechanism in {"unknown", "persistence"}:
            adjustments.append(
                f"mechanism:{mechanism}->repetition:all_same_item"
            )
            mechanism = "repetition"
        if scope != "same item":
            adjustments.append(
                f"scope:{scope}->same item:all_same_item"
            )
            scope = "same item"
        if direction in {"unknown", "stable"}:
            adjustments.append(
                f"direction:{direction}->repeat:all_same_item"
            )
            direction = "repeat"

    evidence = {
        "num_items": len(item_ids),
        "num_unique_item_ids": len(unique_ids),
        "has_exact_item_repeat": bool(has_exact_repeat),
        "all_same_item": bool(all_same_item),
        "same_known_category": bool(same_known_category),
    }

    return mechanism, scope, direction, adjustments, evidence


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# =============================================================================
# Data loading
# =============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_item_info(
    items_meta: Dict[str, Any],
    item_id: Any,
) -> Dict[str, Any]:
    """
    Robust metadata lookup.

    Output IDs are serialized as strings. Missing metadata is marked and the
    interaction is skipped later, matching the original AMem behavior where
    only item IDs found in items_meta are appended to the interaction history.
    """
    sid = str(item_id)
    info = None

    if sid in items_meta:
        info = items_meta[sid]
    elif item_id in items_meta:
        info = items_meta[item_id]
    else:
        try:
            iid = int(item_id)
            if iid in items_meta:
                info = items_meta[iid]
        except (TypeError, ValueError):
            pass

    if info is None:
        return {
            "item_id": sid,
            "item_name": f"Item {sid}",
            "item_category": "Unknown",
            "metadata_found": False,
        }

    category = info.get("main_cat")

    if not category:
        category = info.get("category")

    if not category:
        cats = info.get("categories")
        if isinstance(cats, list) and cats:
            if isinstance(cats[0], list):
                category = " > ".join(str(x) for x in cats[0] if x)
            else:
                category = " > ".join(str(x) for x in cats if x)

    if not category:
        category = "Unknown"

    return {
        "item_id": sid,
        "item_name": str(info.get("title", f"Item {sid}")),
        "item_category": str(category),
        "metadata_found": True,
    }


# =============================================================================
# Window construction
# =============================================================================

def build_windows_for_user(
    user_id: str,
    user_order: int,
    user_data: Dict[str, Any],
    items_meta: Dict[str, Any],
    window_size: int,
    max_train_items: int,
) -> List[Dict[str, Any]]:
    """
    Reproduce the original train_memory() window behavior.

    Original semantics:
        train_items = user_data["train"][-30:]

    A memory is created:
      - whenever current valid interaction count is divisible by window_size
      - OR on the last raw train item
      - if fewer than window_size valid interactions exist, the final shorter
        window is still emitted
      - otherwise the final incomplete chunk becomes the most recent full
        window and can overlap the previous chunk
    """
    train_items = user_data.get("train", [])

    if max_train_items > 0:
        train_items = train_items[-max_train_items:]

    interaction_history: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    window_index = 0

    if not train_items:
        return windows

    for idx, item_id in enumerate(train_items):
        item = get_item_info(items_meta, item_id)

        # Match original code:
        # `if item_id in items_meta: add_interaction(...)`
        if not item["metadata_found"]:
            continue

        interaction_history.append({
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "item_category": item["item_category"],
            "action_type": "purchase",
        })

        current_len = len(interaction_history)
        should_create = False

        if current_len < window_size:
            if idx == len(train_items) - 1 and current_len > 0:
                should_create = True
        else:
            if (
                current_len % window_size == 0
                or idx == len(train_items) - 1
            ):
                should_create = True

        if not should_create:
            continue

        actual_window = min(window_size, current_len)
        window = interaction_history[-actual_window:]

        interaction_summary = [
            {
                "item_id": x["item_id"],
                "item": x["item_name"],
                "category": x["item_category"],
                "action": x["action_type"],
            }
            for x in window
        ]

        windows.append({
            "precompute_id": -1,  # assigned globally later
            "user_id": str(user_id),
            "user_order": int(user_order),
            "window_index": int(window_index),
            "interaction_sequence": [dict(x) for x in window],
            "prompt": build_behavior_extraction_prompt(interaction_summary),
        })
        window_index += 1

    return windows


def build_all_windows(
    user_sequences: Dict[str, Any],
    items_meta: Dict[str, Any],
    number_of_users: int,
    window_size: int,
    max_train_items: int,
) -> List[Dict[str, Any]]:
    user_ids = list(user_sequences.keys())

    if number_of_users > 0:
        user_ids = user_ids[:number_of_users]

    all_windows: List[Dict[str, Any]] = []

    for user_order, user_id in enumerate(
        tqdm(user_ids, desc="Building windows")
    ):
        all_windows.extend(
            build_windows_for_user(
                user_id=str(user_id),
                user_order=user_order,
                user_data=user_sequences[user_id],
                items_meta=items_meta,
                window_size=window_size,
                max_train_items=max_train_items,
            )
        )

    # Stable global order. The later global-memory builder should replay this
    # exact precompute_id order.
    for precompute_id, rec in enumerate(all_windows):
        rec["precompute_id"] = int(precompute_id)

    return all_windows


# =============================================================================
# Batched Unsloth Gemma inference
# =============================================================================

class BatchedGemmaExtractor:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        dtype: str,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for this Unsloth Gemma precompute script."
            )

        if dtype == "auto":
            torch_dtype = None
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "full_finetuning": False,
        }

        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype

        print(f"Loading Gemma with Unsloth FastModel: {model_name}")
        self.model, self.tokenizer = FastModel.from_pretrained(**kwargs)

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            FastModel.for_inference(self.model)
        except Exception:
            # Some Unsloth versions do not expose this method for all models.
            pass

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Gemma device: {self.device}")

    def format_prompt(self, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a behavioral memory modeling system for recommender systems. "
                    "Create both a recommendation-oriented semantic view and a "
                    "domain-agnostic trajectory view. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Matches the working Gemma-3 Unsloth format.
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        return text

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> Tuple[List[str], Dict[str, Any]]:
        rendered = [
            self.format_prompt(prompt)
            for prompt in prompts
        ]

        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # For left-padded batched decoder generation, output[:, input_width:]
        # isolates generated tokens for all rows.
        input_width = int(encoded["input_ids"].shape[1])
        real_input_tokens = int(
            encoded["attention_mask"].sum().item()
        )

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "do_sample": bool(do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs.update({
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            })

        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = self.model.generate(
            **encoded,
            **generate_kwargs,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_ids = outputs[:, input_width:]

        responses = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Approximate generated-token count, sufficient for runtime reporting.
        if self.tokenizer.pad_token_id is not None:
            output_tokens = int(
                (generated_ids != self.tokenizer.pad_token_id)
                .sum()
                .item()
            )
        else:
            output_tokens = int(generated_ids.numel())

        stats = {
            "batch_size": len(prompts),
            "input_tokens": real_input_tokens,
            "output_tokens_approx": output_tokens,
            "elapsed_sec": float(elapsed),
        }

        return responses, stats


# =============================================================================
# JSONL output / resume
# =============================================================================

def inspect_existing_output(
    jsonl_path: Path,
) -> Tuple[Set[int], int]:
    """
    Return completed precompute IDs and valid record count.

    Reject the old schema containing an `embedding` field so resume never
    creates a mixed JSONL with both the old and new formats.
    """
    completed: Set[int] = set()
    valid_records = 0

    if not jsonl_path.exists():
        return completed, valid_records

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # Allow a truncated last line after interruption.
                continue

            if "embedding" in obj:
                raise RuntimeError(
                    f"{jsonl_path} uses the OLD precompute schema containing "
                    f"`embedding` (first detected at line {line_no}). "
                    "Use a new --output filename or delete the old file."
                )

            schema = obj.get("schema_version")
            if schema not in (None, SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported schema_version={schema!r} in {jsonl_path} "
                    f"at line {line_no}."
                )

            if "precompute_id" not in obj:
                raise RuntimeError(
                    f"Missing precompute_id in {jsonl_path} line {line_no}."
                )

            completed.add(int(obj["precompute_id"]))
            valid_records += 1

    return completed, valid_records


def append_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Preserve progress if a long precompute job is interrupted.
        f.flush()
        os.fsync(f.fileno())


def write_run_metadata(
    output_path: Path,
    metadata: Dict[str, Any],
) -> Path:
    meta_path = output_path.with_suffix(
        output_path.suffix + ".meta.json"
    )

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return meta_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute AMem local-memory TEXT with batched Unsloth Gemma. "
            "No embeddings are created or stored."
        )
    )

    # Input / output.
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sequences_file",
        type=str,
        default="user_sequences_10_5000.json",
    )
    parser.add_argument(
        "--items_file",
        type=str,
        default="items.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    # Gemma.
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
    )

    # Memory-window construction.
    parser.add_argument(
        "--number_of_users",
        type=int,
        default=100,
        help="Number of users to precompute; <=0 means all users.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_train_items",
        type=int,
        default=30,
        help="Match original AMem train[-30:]; <=0 means all train items.",
    )

    # Batch generation.
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=8,
    )

    # Greedy by default for reproducible precomputed artifacts.
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume an existing NEW-schema JSONL by skipping completed "
            "precompute_id values."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    sequences_path = data_dir / args.sequences_file
    items_path = data_dir / args.items_file

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("AMem PRECOMPUTE -- LOCAL MEMORY TEXT ONLY")
    print("=" * 80)
    print(f"Sequences       : {sequences_path}")
    print(f"Items           : {items_path}")
    print(f"Output          : {output_path}")
    print(f"Gemma           : {args.model_name}")
    print(f"Users           : {args.number_of_users}")
    print(f"Window size     : {args.window_size}")
    print(f"Max train items : {args.max_train_items}")
    print(f"LLM batch size  : {args.llm_batch_size}")
    print("Embedding model : NONE")
    print("Embedding saved : NO")
    print(f"Schema          : {SCHEMA_VERSION}")

    user_sequences = load_json(sequences_path)
    items_meta = load_json(items_path)

    all_windows = build_all_windows(
        user_sequences=user_sequences,
        items_meta=items_meta,
        number_of_users=args.number_of_users,
        window_size=args.window_size,
        max_train_items=args.max_train_items,
    )

    print(
        f"Prepared {len(all_windows)} local-memory windows"
    )

    # Output behavior:
    # - resume=True: inspect and skip completed IDs
    # - resume=False: start clean instead of accidentally appending duplicates
    if args.resume:
        completed_ids, existing_valid_records = inspect_existing_output(
            output_path
        )
    else:
        completed_ids = set()
        existing_valid_records = 0

        if output_path.exists():
            print(
                f"Resume disabled: truncating existing output {output_path}"
            )
            output_path.unlink()

        old_meta = output_path.with_suffix(
            output_path.suffix + ".meta.json"
        )
        if old_meta.exists():
            old_meta.unlink()

    if completed_ids:
        print(
            f"Resume: {len(completed_ids)} completed precompute IDs "
            f"({existing_valid_records} valid JSONL records)"
        )

    pending = [
        rec
        for rec in all_windows
        if rec["precompute_id"] not in completed_ids
    ]

    if not pending:
        print("Nothing to do: all local memories are already precomputed.")
        return

    print(
        f"Pending local memories: {len(pending)}"
    )

    extractor = BatchedGemmaExtractor(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
    )

    total_llm_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    parse_failures = 0
    consistency_adjustment_records = 0
    consistency_adjustment_events = 0
    produced = 0

    batch_starts = range(
        0,
        len(pending),
        args.llm_batch_size,
    )

    progress = tqdm(
        batch_starts,
        desc="Gemma batches",
    )

    for start in progress:
        batch = pending[
            start:start + args.llm_batch_size
        ]

        prompts = [
            record["prompt"]
            for record in batch
        ]

        responses, batch_stats = extractor.generate_batch(
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.sample,
        )

        total_llm_time += batch_stats["elapsed_sec"]
        total_input_tokens += batch_stats["input_tokens"]
        total_output_tokens += batch_stats["output_tokens_approx"]

        output_records: List[Dict[str, Any]] = []

        for rec, response in zip(
            batch,
            responses,
        ):
            try:
                parsed = parse_json_response(response)

                behavior_explanation = str(
                    parsed.get(
                        "behavior_explanation",
                        "",
                    )
                ).strip()

                recommendation_behavior = str(
                    parsed.get(
                        "recommendation_behavior",
                        "",
                    )
                ).strip()

                semantic_focus = parsed.get(
                    "semantic_focus",
                    [],
                )
                if not isinstance(
                    semantic_focus,
                    list,
                ):
                    semantic_focus = [
                        str(semantic_focus)
                    ] if semantic_focus else []

                semantic_focus = [
                    str(x).strip()
                    for x in semantic_focus
                    if str(x).strip()
                ][:8]

                trajectory_signature = str(
                    parsed.get(
                        "trajectory_signature",
                        "",
                    )
                ).strip()

                mechanism = str(
                    parsed.get(
                        "mechanism",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                scope = str(
                    parsed.get(
                        "scope",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                direction = str(
                    parsed.get(
                        "direction",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                try:
                    confidence = float(
                        parsed.get(
                            "confidence",
                            0.0,
                        )
                    )
                except Exception:
                    confidence = 0.0

                confidence = max(
                    0.0,
                    min(1.0, confidence),
                )

                (
                    mechanism,
                    scope,
                    direction,
                    consistency_adjustments,
                    structural_evidence,
                ) = enforce_structural_consistency(
                    mechanism=mechanism,
                    scope=scope,
                    direction=direction,
                    interaction_sequence=rec[
                        "interaction_sequence"
                    ],
                )

                if not recommendation_behavior:
                    raise ValueError(
                        "Missing recommendation_behavior"
                    )

                if not trajectory_signature:
                    raise ValueError(
                        "Missing trajectory_signature"
                    )

                parse_ok = True
                parse_error = None

            except Exception as exc:
                parse_failures += 1

                # Conservative fallback: retain observed category evidence but
                # mark the trajectory abstraction as unknown.
                categories = [
                    str(
                        x.get(
                            "item_category",
                            "Unknown",
                        )
                    ).strip()
                    for x in rec[
                        "interaction_sequence"
                    ]
                    if str(
                        x.get(
                            "item_category",
                            "",
                        )
                    ).strip()
                ]

                unique_categories = list(
                    dict.fromkeys(categories)
                )

                behavior_explanation = (
                    f"User purchased "
                    f"{len(rec['interaction_sequence'])} observed items."
                )

                if unique_categories:
                    recommendation_behavior = (
                        "Observed preference is concentrated around: "
                        + ", ".join(
                            unique_categories[:4]
                        )
                        + "."
                    )
                else:
                    recommendation_behavior = (
                        "Observed preference could not be reliably summarized."
                    )

                semantic_focus = (
                    unique_categories[:8]
                )
                trajectory_signature = (
                    "uncertain observed behavior"
                )
                mechanism = "unknown"
                scope = "unknown"
                direction = "unknown"
                confidence = 0.0
                consistency_adjustments = []
                structural_evidence = {
                    "num_items": len(
                        rec["interaction_sequence"]
                    ),
                    "num_unique_item_ids": len({
                        str(x.get("item_id"))
                        for x in rec[
                            "interaction_sequence"
                        ]
                    }),
                    "has_exact_item_repeat": False,
                    "all_same_item": False,
                    "same_known_category": False,
                }

                parse_ok = False
                parse_error = repr(exc)

            # -----------------------------------------------------------------
            # Backward-compatibility aliases
            # -----------------------------------------------------------------
            # Existing AMem/global-memory code often expects:
            #   pattern_description + keywords
            # Existing Tree code often expects:
            #   behavior_signature
            #
            # Keep those names, but give them explicit roles:
            #   pattern_description = recommendation-oriented semantic view
            #   keywords            = semantic_focus
            #   behavior_signature  = trajectory_signature
            pattern_description = (
                recommendation_behavior
            )
            keywords = list(
                semantic_focus
            )
            behavior_signature = (
                trajectory_signature
            )

            output_record = {
                "schema_version": SCHEMA_VERSION,
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
                ),
                "precompute_id": int(
                    rec["precompute_id"]
                ),
                "user_id": rec["user_id"],
                "user_order": int(
                    rec["user_order"]
                ),
                "window_index": int(
                    rec["window_index"]
                ),
                "interaction_sequence": (
                    rec["interaction_sequence"]
                ),

                # Grounded explanation.
                "behavior_explanation": (
                    behavior_explanation
                ),

                # View A: recommendation/ranking semantics.
                "recommendation_behavior": (
                    recommendation_behavior
                ),
                "semantic_focus": (
                    semantic_focus
                ),

                # View B: cross-user trajectory abstraction.
                "trajectory_signature": (
                    trajectory_signature
                ),
                "behavior_signature": (
                    behavior_signature
                ),
                "mechanism": mechanism,
                "scope": scope,
                "direction": direction,
                "confidence": confidence,
                "consistency_adjustments": (
                    consistency_adjustments
                ),
                "structural_evidence": (
                    structural_evidence
                ),

                # Backward-compatible aliases used by existing AMem code.
                "pattern_description": (
                    pattern_description
                ),
                "keywords": keywords,

                "parse_ok": bool(parse_ok),
                "parse_error": parse_error,

                # Useful for auditing/debugging Gemma extraction.
                "raw_response": response,
            }

            if consistency_adjustments:
                consistency_adjustment_records += 1
                consistency_adjustment_events += len(
                    consistency_adjustments
                )

            # Intentionally NO "embedding" field.
            output_records.append(
                output_record
            )

        append_jsonl(
            output_path,
            output_records,
        )

        produced += len(output_records)

        progress.set_postfix({
            "new": produced,
            "parse_fail": parse_failures,
            "sec/batch": (
                f"{batch_stats['elapsed_sec']:.2f}"
            ),
        })

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "behavior_prompt_version": BEHAVIOR_PROMPT_VERSION,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
        "stage": "local_memory_text_precompute",
        "contains_embeddings": False,
        "sequences_file": str(sequences_path),
        "items_file": str(items_path),
        "output_file": str(output_path),
        "model_name": args.model_name,
        "number_of_users": args.number_of_users,
        "window_size": args.window_size,
        "max_train_items": args.max_train_items,
        "llm_batch_size": args.llm_batch_size,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
        "sample": args.sample,
        "temperature": (
            args.temperature if args.sample else None
        ),
        "top_p": (
            args.top_p if args.sample else None
        ),
        "top_k": (
            args.top_k if args.sample else None
        ),
        "seed": args.seed,
        "num_windows_total": len(all_windows),
        "num_newly_produced": produced,
        "num_previously_completed": len(
            completed_ids
        ),
        "parse_failures_this_run": (
            parse_failures
        ),
        "consistency_adjustment_records": int(
            consistency_adjustment_records
        ),
        "consistency_adjustment_events": int(
            consistency_adjustment_events
        ),
        "llm_inference_time_sec": round(
            total_llm_time,
            4,
        ),
        "llm_input_tokens": (
            total_input_tokens
        ),
        "llm_output_tokens_approx": (
            total_output_tokens
        ),
    }

    metadata_path = write_run_metadata(
        output_path,
        metadata,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output JSONL    : {output_path}")
    print(f"Metadata        : {metadata_path}")
    print(f"New records     : {produced}")
    print(f"Parse failures  : {parse_failures}")
    print(
        f"Consistency fix : {consistency_adjustment_records} records / "
        f"{consistency_adjustment_events} label adjustments"
    )
    print(f"Gemma time      : {total_llm_time:.2f} sec")
    print("Embeddings      : NOT computed / NOT stored")
    print("\nNext stage:")
    print(
        "load this JSONL -> compose memory texts -> batch-encode embeddings "
        "once -> sequential cosine/link/evolve/store."
    )


if __name__ == "__main__":
    main()
