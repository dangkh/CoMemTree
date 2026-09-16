#!/usr/bin/env python3
"""
Oracle Next-Behavior diagnostic for recommendation.

Goal
----
Test whether a CORRECT next-behavior description is actually useful for
selecting the ground-truth item from the CURRENT candidate pool.

For each test user:
    current train history + current ground-truth test item
        -> Gemma generates ORACLE next behavior
        -> Gemma ranks the SAME current candidate pool

The script also runs a Native-LLM baseline on the exact same history and
candidate ordering, so the comparison is controlled.

IMPORTANT
---------
This is an ORACLE / upper-bound diagnostic, NOT a valid deployable recommender:
the ground-truth test item is intentionally used to construct the oracle behavior.

Data format matches the current CDs pipeline:
  items.json
  user_sequences_10_5000.json
  user_negatives_10_5000.json

Candidate construction matches inference_flat_faiss_gemma.py:
  candidates = test_ids + test_neg
  shuffle with random.Random(f"{seed}:{user_id}")

Outputs
-------
<output>.json
    per-user native and oracle rankings

<output>_summary.json
    Hit/NDCG comparison and rank deltas

<output>_oracle_behaviors.jsonl
    reusable cache of generated oracle behaviors

Example
-------
python oracle_behavior_ranking_gemma.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --output results/oracle_behavior_cd_300.json \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --number-of-users 300 \
  --history-size 10 \
  --batch-size 8 \
  --behavior-max-new-tokens 192 \
  --rank-max-new-tokens 768 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


# =============================================================================
# Basic utilities
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


def save_json_atomic(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_items(items: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in items.items()}


def get_item_info(item_id: Any, items_meta: Dict[str, Any]) -> Dict[str, str]:
    iid = str(item_id)
    info = items_meta.get(iid, {})

    category = (
        info.get("main_cat")
        or info.get("category")
        or info.get("categories")
        or "Unknown"
    )
    if isinstance(category, list):
        category = " > ".join(map(str, category[:5]))

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


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def extract_json_value(text: str) -> Any:
    """Robustly recover the first complete JSON object/array."""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    candidates = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        pos = s.find(opener)
        if pos >= 0:
            candidates.append((pos, opener, closer))

    if not candidates:
        raise ValueError("No JSON object/array found")

    start, opener, closer = min(candidates, key=lambda z: z[0])
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


def sanitize_ranking(predicted: Any, valid_ids: Sequence[str]) -> Tuple[List[str], bool]:
    valid = [str(x) for x in valid_ids]
    valid_set = set(valid)

    if not isinstance(predicted, list):
        predicted = []

    out: List[str] = []
    seen = set()
    for x in predicted:
        iid = str(x)
        if iid in valid_set and iid not in seen:
            out.append(iid)
            seen.add(iid)

    parse_complete = (len(out) == len(valid))

    # Safe fallback: append missing candidates in their ORIGINAL candidate order.
    for iid in valid:
        if iid not in seen:
            out.append(iid)
            seen.add(iid)

    return out, parse_complete


# =============================================================================
# Metrics
# =============================================================================

def target_rank_1based(ranking: Sequence[str], target: str) -> int:
    target = str(target)
    try:
        return list(map(str, ranking)).index(target) + 1
    except ValueError:
        return len(ranking) + 1


def metrics_from_ranks(ranks: List[int], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    if not ranks:
        return {}

    out: Dict[str, float] = {}
    n = len(ranks)

    for k in ks:
        hit = sum(r <= k for r in ranks) / n
        ndcg = sum(
            (1.0 / math.log2(r + 1)) if r <= k else 0.0
            for r in ranks
        ) / n
        out[f"Hit@{k}"] = float(hit)
        out[f"NDCG@{k}"] = float(ndcg)

    out["MRR"] = float(sum(1.0 / r for r in ranks) / n)
    out["MeanRank"] = float(sum(ranks) / n)
    out["MedianRank"] = float(np.median(ranks))
    return out


# =============================================================================
# Oracle behavior handling
# =============================================================================

ALLOWED_MECHANISMS = [
    "repetition",
    "persistence",
    "deepening",
    "collection expansion",
    "adjacent exploration",
    "cross category exploration",
    "broadening",
    "switching",
    "other",
]

ALLOWED_SCOPES = [
    "same item",
    "same creator",
    "same series",
    "same subcategory",
    "same category",
    "adjacent category",
    "cross category",
    "other",
]

ALLOWED_DIRECTIONS = [
    "repeat",
    "stable",
    "deepen",
    "narrow",
    "broaden",
    "shift",
    "mixed",
    "other",
]


def behavior_prompt(
    history_items: List[Dict[str, str]],
    target_item: Dict[str, str],
) -> str:
    """
    The GT item is intentionally visible here because this is an oracle diagnostic.

    We explicitly forbid direct title/item-id leakage in the returned behavior.
    Rich category/semantic behavior is allowed because that is exactly what
    this experiment is testing.
    """
    return f"""You are constructing an ORACLE next-behavior description for a recommendation diagnostic.

You are given:
1. the user's recent observed history;
2. the item the user actually selected next.

Infer the latent NEXT BEHAVIOR that best explains the transition from the recent history to the next selected item.

The output should be USEFUL for identifying what kind of item the user is likely to choose next, but it must remain a reusable behavior description rather than revealing the answer directly.

STRICT RULES:
- Do NOT output the target item_id.
- Do NOT copy or mention the exact target title.
- Do NOT say "ground truth", "correct item", "target item", or candidate position.
- Do NOT simply paraphrase the target title.
- You MAY use abstract semantic/category information implied by the target.
- The next_behavior_description should be specific enough to distinguish plausible next choices from unrelated candidates.
- Describe the transition from recent history to the next choice.

Recent history (oldest to newest):
{json.dumps(history_items, ensure_ascii=False, indent=2)}

Actual next selected item:
{json.dumps(target_item, ensure_ascii=False, indent=2)}

Use one of these broad mechanism labels when possible:
{json.dumps(ALLOWED_MECHANISMS, ensure_ascii=False)}

Use one of these scope labels when possible:
{json.dumps(ALLOWED_SCOPES, ensure_ascii=False)}

Use one of these direction labels when possible:
{json.dumps(ALLOWED_DIRECTIONS, ensure_ascii=False)}

Return JSON only:
{{
  "behavior_signature": "short abstract phrase",
  "mechanism": "one broad label",
  "scope": "one broad label",
  "direction": "one broad label",
  "next_behavior_description": "one concise but semantically informative sentence"
}}
"""


def normalize_oracle_behavior(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, dict):
        raise ValueError("Oracle behavior output is not an object")

    out = {
        "behavior_signature": clean_text(obj.get("behavior_signature")),
        "mechanism": clean_text(obj.get("mechanism")).lower(),
        "scope": clean_text(obj.get("scope")).lower(),
        "direction": clean_text(obj.get("direction")).lower(),
        "next_behavior_description": clean_text(obj.get("next_behavior_description")),
    }

    if not out["next_behavior_description"]:
        raise ValueError("Missing next_behavior_description")

    if not out["behavior_signature"]:
        out["behavior_signature"] = out["next_behavior_description"][:120]

    return out


def leakage_flags(
    behavior: Dict[str, str],
    target_item: Dict[str, str],
) -> Dict[str, bool]:
    blob = " ".join(str(v) for v in behavior.values()).lower()
    iid = str(target_item["item_id"]).lower().strip()
    title = clean_text(target_item["title"]).lower()

    # Exact title check only when title is meaningful.
    title_leak = bool(
        title
        and not title.startswith("item ")
        and len(title) >= 5
        and title in blob
    )
    id_leak = bool(iid and iid in blob)

    return {
        "target_item_id_leak": id_leak,
        "exact_target_title_leak": title_leak,
        "any_direct_leak": id_leak or title_leak,
    }


# =============================================================================
# Ranking prompts
# =============================================================================

def native_rank_prompt(
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
) -> str:
    ids = [x["item_id"] for x in candidate_items]
    return f"""You are a recommendation ranking system.

Rank ALL candidate items for the target user.

Priority:
1. Recent observed history and recency.
2. Semantic/category compatibility with the user's recent preferences.

Recent history (most recent last):
{json.dumps(history_items, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidate_items, ensure_ascii=False, indent=2)}

Requirements:
- Rank ALL {len(ids)} candidate IDs exactly once.
- Do not invent or omit IDs.
- Do not preserve input order by default.
- Return JSON only.

{{"ranked_item_ids":[...],"reasoning":"one concise sentence"}}
"""


def oracle_rank_prompt(
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    oracle_behavior: Dict[str, str],
) -> str:
    ids = [x["item_id"] for x in candidate_items]

    # Deliberately expose ONLY the generated behavior, never the GT item.
    behavior_view = {
        "behavior_signature": oracle_behavior["behavior_signature"],
        "mechanism": oracle_behavior["mechanism"],
        "scope": oracle_behavior["scope"],
        "direction": oracle_behavior["direction"],
        "next_behavior_description": oracle_behavior["next_behavior_description"],
    }

    return f"""You are a recommendation ranking system.

Rank ALL candidate items for the target user.

Priority:
1. Use the provided NEXT BEHAVIOR to identify the most plausible NEXT choice.
2. Check that the candidate is compatible with the user's recent observed history.
3. Use semantic/category fit to distinguish candidates.

The NEXT BEHAVIOR is an auxiliary prediction of what transition the user is about to make.
It describes the NEXT choice, not merely the user's current state.

Recent history (most recent last):
{json.dumps(history_items, ensure_ascii=False, indent=2)}

Predicted NEXT BEHAVIOR:
{json.dumps(behavior_view, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidate_items, ensure_ascii=False, indent=2)}

Requirements:
- Rank ALL {len(ids)} candidate IDs exactly once.
- Use the NEXT BEHAVIOR actively when distinguishing plausible candidates.
- Do not invent or omit IDs.
- Do not preserve input order by default.
- Return JSON only.

{{"ranked_item_ids":[...],"reasoning":"one concise sentence"}}
"""


# =============================================================================
# Local Gemma
# =============================================================================

class LocalGemmaBatch:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("This Unsloth Gemma script expects a CUDA GPU.")

        print(f"[INFO] Loading model: {model_name}", flush=True)

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            full_finetuning=False,
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_seq_length = int(max_seq_length)

        self.calls = 0
        self.examples = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_time = 0.0

    def _render(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if text.startswith("<bos>"):
            text = text[len("<bos>"):]
        return text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        if not prompts:
            return []

        rendered = [self._render(p) for p in prompts]

        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        input_len = int(batch["input_ids"].shape[1])
        n_input_tokens = int(batch["attention_mask"].sum().item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        decoded: List[str] = []
        out_tok = 0

        for row in outputs:
            gen = row[input_len:]
            # Count non-pad tokens approximately.
            n = int((gen != self.tokenizer.pad_token_id).sum().item())
            out_tok += n
            decoded.append(
                self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            )

        self.calls += 1
        self.examples += len(prompts)
        self.input_tokens += n_input_tokens
        self.output_tokens += out_tok
        self.total_time += elapsed

        return decoded

    def stats(self) -> Dict[str, Any]:
        return {
            "generate_calls": self.calls,
            "examples": self.examples,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_time_sec": round(self.total_time, 3),
            "examples_per_second": (
                round(self.examples / self.total_time, 4)
                if self.total_time > 0
                else 0.0
            ),
        }


# =============================================================================
# Cache
# =============================================================================

def load_behavior_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("oracle_ok") is True:
                    out[str(row["user_id"])] = row
            except Exception:
                continue
    return out


# =============================================================================
# Per-user data construction
# =============================================================================

def build_user_request(
    uid: str,
    user_data: Dict[str, Any],
    neg_data: Dict[str, Any],
    items_meta: Dict[str, Any],
    history_size: int,
    seed: int,
) -> Dict[str, Any]:
    train_ids = [str(x) for x in user_data.get("train", [])]
    test_ids = [str(x) for x in user_data.get("test", [])]
    neg_ids = [str(x) for x in neg_data.get("test_neg", [])]

    if not test_ids:
        raise ValueError(f"user={uid}: no test item")

    # Current pipeline generally has one held-out test item.
    # Use the first as the diagnostic target while keeping ALL test ids in candidate pool.
    target_id = test_ids[0]

    history_items = [
        get_item_info(i, items_meta)
        for i in train_ids[-int(history_size):]
    ]

    target_item = get_item_info(target_id, items_meta)

    candidate_ids = test_ids + neg_ids
    rng = random.Random(f"{seed}:{uid}")
    rng.shuffle(candidate_ids)

    candidate_items = [
        get_item_info(i, items_meta)
        for i in candidate_ids
    ]

    return {
        "user_id": str(uid),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "target_id": target_id,
        "history_items": history_items,
        "target_item": target_item,
        "candidate_ids": candidate_ids,
        "candidate_items": candidate_items,
    }


# =============================================================================
# Main experiment
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle next-behavior upper-bound diagnostic."
    )

    p.add_argument("--items", required=True)
    p.add_argument("--sequences", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--output", required=True)

    p.add_argument(
        "--model",
        default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    )
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--number-of-users", type=int, default=300)
    p.add_argument("--start-user", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--behavior-max-new-tokens", type=int, default=192)
    p.add_argument("--rank-max-new-tokens", type=int, default=768)

    p.add_argument(
        "--run-native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run the controlled Native-LLM baseline.",
    )
    p.add_argument(
        "--reject-direct-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject oracle behaviors that copy exact target item_id/title.",
    )

    p.add_argument(
        "--oracle-cache",
        default=None,
        help=(
            "Optional JSONL behavior cache. Default: "
            "<output_stem>_oracle_behaviors.jsonl"
        ),
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    for path in [args.items, args.sequences, args.negatives]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    items_meta = normalize_items(load_json(args.items))
    sequences_raw = load_json(args.sequences)
    negatives_raw = load_json(args.negatives)

    sequences = {str(k): v for k, v in sequences_raw.items()}
    negatives = {str(k): v for k, v in negatives_raw.items()}

    all_users = list(sequences.keys())
    users = all_users[int(args.start_user):]

    if int(args.number_of_users) > 0:
        users = users[:int(args.number_of_users)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output.with_name(output.stem + "_summary.json")
    cache_path = (
        Path(args.oracle_cache)
        if args.oracle_cache
        else output.with_name(output.stem + "_oracle_behaviors.jsonl")
    )

    requests: Dict[str, Dict[str, Any]] = {}
    for uid in users:
        requests[uid] = build_user_request(
            uid=uid,
            user_data=sequences[uid],
            neg_data=negatives.get(uid, {}),
            items_meta=items_meta,
            history_size=args.history_size,
            seed=args.seed,
        )

    print("=" * 88)
    print("ORACLE NEXT-BEHAVIOR DIAGNOSTIC")
    print("=" * 88)
    print(f"users                : {len(users)}")
    print(f"history size         : {args.history_size}")
    print(f"model                : {args.model}")
    print(f"oracle cache         : {cache_path}")
    print(f"run native baseline  : {args.run_native}")
    print(f"reject direct leakage: {args.reject_direct_leak}")

    llm = LocalGemmaBatch(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    # -------------------------------------------------------------------------
    # Phase 1: Oracle behavior generation
    # -------------------------------------------------------------------------
    behavior_cache = load_behavior_cache(cache_path)

    need_behavior = [
        uid for uid in users
        if uid not in behavior_cache
    ]

    print(f"\n[1/3] Oracle behavior generation: pending={len(need_behavior)}")

    pbar = tqdm(total=len(need_behavior), desc="oracle behavior")

    for start in range(0, len(need_behavior), args.batch_size):
        batch_users = need_behavior[start:start + args.batch_size]

        prompts = [
            behavior_prompt(
                requests[uid]["history_items"],
                requests[uid]["target_item"],
            )
            for uid in batch_users
        ]

        outputs = llm.generate_batch(
            prompts,
            max_new_tokens=args.behavior_max_new_tokens,
        )

        for uid, raw in zip(batch_users, outputs):
            req = requests[uid]
            row: Dict[str, Any]

            try:
                obj = extract_json_value(raw)
                behavior = normalize_oracle_behavior(obj)
                flags = leakage_flags(behavior, req["target_item"])

                oracle_ok = True
                rejection_reason = None

                if args.reject_direct_leak and flags["any_direct_leak"]:
                    oracle_ok = False
                    rejection_reason = "direct_target_leak"

                row = {
                    "user_id": uid,
                    "oracle_ok": oracle_ok,
                    "rejection_reason": rejection_reason,
                    "target_id": req["target_id"],
                    "target_item": req["target_item"],
                    "history_item_ids": [
                        x["item_id"] for x in req["history_items"]
                    ],
                    "oracle_behavior": behavior,
                    "leakage_flags": flags,
                    "raw_output": raw,
                }
            except Exception as e:
                row = {
                    "user_id": uid,
                    "oracle_ok": False,
                    "rejection_reason": f"parse_error:{type(e).__name__}:{e}",
                    "target_id": req["target_id"],
                    "target_item": req["target_item"],
                    "history_item_ids": [
                        x["item_id"] for x in req["history_items"]
                    ],
                    "oracle_behavior": None,
                    "leakage_flags": {},
                    "raw_output": raw,
                }

            append_jsonl(cache_path, row)
            if row["oracle_ok"]:
                behavior_cache[uid] = row

        pbar.update(len(batch_users))

    pbar.close()

    usable_users = [
        uid for uid in users
        if uid in behavior_cache
        and behavior_cache[uid].get("oracle_ok") is True
    ]

    print(
        f"      usable oracle behaviors={len(usable_users)}/{len(users)} "
        f"({len(usable_users)/len(users):.3f})"
        if users else "      no users"
    )

    if not usable_users:
        raise RuntimeError("No usable oracle behaviors were generated.")

    # -------------------------------------------------------------------------
    # Resume ranking results
    # -------------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    completed = set()

    if args.resume and output.exists():
        try:
            old = load_json(output)
            if isinstance(old, list):
                results = old
                completed = {str(x["user_id"]) for x in results}
                print(f"Resume ranking: {len(completed)} completed users")
        except Exception as e:
            print(f"[WARN] Could not resume ranking output: {e}")
            results = []
            completed = set()

    pending = [uid for uid in usable_users if uid not in completed]

    # -------------------------------------------------------------------------
    # Phase 2: Controlled Native + Oracle ranking
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Ranking: pending={len(pending)}")
    pbar = tqdm(total=len(pending), desc="oracle ranking")
    processed_since_save = 0

    for start in range(0, len(pending), args.batch_size):
        batch_users = pending[start:start + args.batch_size]

        native_outputs: List[Optional[str]] = [None] * len(batch_users)

        if args.run_native:
            native_prompts = [
                native_rank_prompt(
                    requests[uid]["history_items"],
                    requests[uid]["candidate_items"],
                )
                for uid in batch_users
            ]
            native_outputs = llm.generate_batch(
                native_prompts,
                max_new_tokens=args.rank_max_new_tokens,
            )

        oracle_prompts = [
            oracle_rank_prompt(
                requests[uid]["history_items"],
                requests[uid]["candidate_items"],
                behavior_cache[uid]["oracle_behavior"],
            )
            for uid in batch_users
        ]
        oracle_outputs = llm.generate_batch(
            oracle_prompts,
            max_new_tokens=args.rank_max_new_tokens,
        )

        batch_rows: List[Dict[str, Any]] = []

        for idx, uid in enumerate(batch_users):
            req = requests[uid]
            valid_ids = req["candidate_ids"]

            # Native
            native_ranking: Optional[List[str]] = None
            native_reasoning = ""
            native_parse_complete: Optional[bool] = None

            if args.run_native:
                raw_native = native_outputs[idx] or ""
                try:
                    obj = extract_json_value(raw_native)
                    native_ranking, native_parse_complete = sanitize_ranking(
                        obj.get("ranked_item_ids", []),
                        valid_ids,
                    )
                    native_reasoning = clean_text(obj.get("reasoning"))
                except Exception:
                    native_ranking = list(valid_ids)
                    native_parse_complete = False

            # Oracle
            raw_oracle = oracle_outputs[idx]
            try:
                obj = extract_json_value(raw_oracle)
                oracle_ranking, oracle_parse_complete = sanitize_ranking(
                    obj.get("ranked_item_ids", []),
                    valid_ids,
                )
                oracle_reasoning = clean_text(obj.get("reasoning"))
            except Exception:
                oracle_ranking = list(valid_ids)
                oracle_parse_complete = False
                oracle_reasoning = ""

            target = req["target_id"]
            oracle_rank = target_rank_1based(oracle_ranking, target)

            if native_ranking is not None:
                native_rank = target_rank_1based(native_ranking, target)
                rank_delta = native_rank - oracle_rank
            else:
                native_rank = None
                rank_delta = None

            row = {
                "user_id": uid,
                "target": target,
                "ground_truth_item_ids": req["test_ids"],
                "candidate_item_ids": valid_ids,
                "candidate_items": req["candidate_items"],
                "history_items": req["history_items"],
                "oracle_behavior": behavior_cache[uid]["oracle_behavior"],
                "oracle_leakage_flags": behavior_cache[uid].get("leakage_flags", {}),
                "oracle_ranking": oracle_ranking,
                "oracle_rank_position": oracle_rank,
                "oracle_parse_complete": oracle_parse_complete,
                "oracle_reasoning": oracle_reasoning,
                "native_ranking": native_ranking,
                "native_rank_position": native_rank,
                "native_parse_complete": native_parse_complete,
                "native_reasoning": native_reasoning,
                # Positive means Oracle improved the rank.
                "rank_improvement_native_minus_oracle": rank_delta,
            }
            batch_rows.append(row)

        results.extend(batch_rows)
        processed_since_save += len(batch_rows)
        pbar.update(len(batch_users))

        if args.save_every > 0 and processed_since_save >= args.save_every:
            save_json_atomic(results, output)
            processed_since_save = 0

    pbar.close()
    save_json_atomic(results, output)

    # -------------------------------------------------------------------------
    # Phase 3: Summary
    # -------------------------------------------------------------------------
    print("\n[3/3] Summary")

    selected = [
        r for r in results
        if str(r["user_id"]) in set(usable_users)
    ]

    oracle_ranks = [int(r["oracle_rank_position"]) for r in selected]
    oracle_metrics = metrics_from_ranks(oracle_ranks)

    native_metrics = None
    native_ranks: List[int] = []

    if args.run_native:
        native_ranks = [
            int(r["native_rank_position"])
            for r in selected
            if r.get("native_rank_position") is not None
        ]
        native_metrics = metrics_from_ranks(native_ranks)

    improvements = [
        int(r["rank_improvement_native_minus_oracle"])
        for r in selected
        if r.get("rank_improvement_native_minus_oracle") is not None
    ]

    metric_gain: Dict[str, float] = {}
    if native_metrics is not None:
        for k, v in oracle_metrics.items():
            if k in native_metrics:
                # For MeanRank/MedianRank lower is better; keep raw Oracle-Native separately.
                if k in {"MeanRank", "MedianRank"}:
                    metric_gain[k] = float(native_metrics[k] - v)
                else:
                    metric_gain[k] = float(v - native_metrics[k])

    summary = {
        "experiment": "oracle_next_behavior_upper_bound",
        "warning": (
            "Ground-truth test item is intentionally used to generate oracle behavior. "
            "This is a diagnostic upper bound, not deployable recommendation performance."
        ),
        "model": args.model,
        "num_requested_users": len(users),
        "num_usable_oracle_users": len(usable_users),
        "oracle_behavior_coverage": (
            len(usable_users) / len(users) if users else 0.0
        ),
        "same_current_test_users": True,
        "same_current_ground_truth_items": True,
        "same_current_candidate_pool": True,
        "candidate_construction": "test_ids + test_neg; stable shuffle with seed:user_id",
        "history_size": args.history_size,
        "oracle_metrics": oracle_metrics,
        "native_metrics": native_metrics,
        "oracle_minus_native_metric_gain": metric_gain,
        "rank_improvement": {
            "definition": "native_rank - oracle_rank; positive means Oracle is better",
            "mean": float(np.mean(improvements)) if improvements else None,
            "median": float(np.median(improvements)) if improvements else None,
            "users_improved": int(sum(x > 0 for x in improvements)),
            "users_unchanged": int(sum(x == 0 for x in improvements)),
            "users_worsened": int(sum(x < 0 for x in improvements)),
            "improved_ratio": (
                float(sum(x > 0 for x in improvements) / len(improvements))
                if improvements else None
            ),
        },
        "parse_quality": {
            "oracle_complete_ratio": (
                sum(bool(r.get("oracle_parse_complete")) for r in selected) / len(selected)
                if selected else 0.0
            ),
            "native_complete_ratio": (
                sum(bool(r.get("native_parse_complete")) for r in selected) / len(selected)
                if selected and args.run_native else None
            ),
        },
        "files": {
            "results": str(output),
            "oracle_behavior_cache": str(cache_path),
            "summary": str(summary_path),
        },
        "llm_stats": llm.stats(),
        "config": vars(args),
    }

    save_json_atomic(summary, summary_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
