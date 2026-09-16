#!/usr/bin/env python3
"""
Convert GroupLens MovieLens Latest into the three JSON inputs used by AMem4Rec.

Default output filenames:
  items.json
  user_sequences_10_5000.json
  user_negatives_10_5000.json

Recovered project preprocessing semantics:
  - one latest interaction per (user, item)
  - sort each user's interactions chronologically
  - leave-two-out: train = items[:-2], val = [items[-2]], test = [items[-1]]
  - 19 popularity-biased negatives for validation and test independently
  - negatives exclude every item ever interacted with by that user
  - seed 42

Interpretation used for the current *_10_5000 filename:
  - min unique interactions/user = 10
  - max unique interactions/user = 5000
This is configurable with --min-interactions / --max-interactions.
The number of evaluation/precompute users remains a downstream setting and is NOT
hard-capped by this converter.

MovieLens mapping:
  item_id   <- movieId
  title     <- movies.csv:title
  main_cat  <- movies.csv:genres (pipe-separated string)
  category  <- same as main_cat (extra compatibility)
  genres    <- list form of genres

All MovieLens ratings are treated as implicit interactions by default, mirroring the
existing project pipeline. Use --min-rating only if you intentionally want to change
that protocol.

The implementation is streaming and makes two passes through ratings.csv, so it can
process the full ~33M-rating ml-latest dataset without loading it into pandas memory.
Input may be an extracted ml-latest directory or ml-latest.zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import numpy as np


DEFAULT_MIN_INTERACTIONS = 10
DEFAULT_MAX_INTERACTIONS = 5000
DEFAULT_NEG_NUM = 19
DEFAULT_SEED = 42


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare MovieLens Latest for AMem4Rec")
    p.add_argument(
        "--input",
        required=True,
        help="Path to extracted ml-latest directory OR ml-latest.zip",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the three project JSON files are written",
    )
    p.add_argument(
        "--min-interactions",
        type=int,
        default=DEFAULT_MIN_INTERACTIONS,
        help="Minimum unique interactions per user. Default: 10",
    )
    p.add_argument(
        "--max-interactions",
        type=int,
        default=DEFAULT_MAX_INTERACTIONS,
        help="Maximum unique interactions per user; <=0 disables upper bound. Default: 5000",
    )
    p.add_argument(
        "--neg-num",
        type=int,
        default=DEFAULT_NEG_NUM,
        help="Popularity-biased negatives for validation and test. Default: 19",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--min-rating",
        type=float,
        default=None,
        help="Optional positive-rating threshold. Default: None (all ratings are interactions)",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON",
    )
    p.add_argument(
        "--limit-users",
        type=int,
        default=0,
        help=(
            "Debug-only output cap after filtering; <=0 keeps all valid users. "
            "Popularity is still estimated from ALL valid users."
        ),
    )
    return p.parse_args()


# =============================================================================
# MovieLens source abstraction: extracted directory OR ZIP
# =============================================================================

class MovieLensSource:
    def __init__(self, src: Path):
        self.src = src
        self._zip: Optional[zipfile.ZipFile] = None
        if src.is_file() and src.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(src, "r")
            names = self._zip.namelist()
            self._movies_name = self._find_member(names, "movies.csv")
            self._ratings_name = self._find_member(names, "ratings.csv")
        elif src.is_dir():
            self._movies_path = self._find_path(src, "movies.csv")
            self._ratings_path = self._find_path(src, "ratings.csv")
        else:
            raise FileNotFoundError(f"Input is neither a ZIP nor a directory: {src}")

    @staticmethod
    def _find_member(names: Sequence[str], basename: str) -> str:
        hits = [n for n in names if Path(n).name == basename]
        if not hits:
            raise FileNotFoundError(f"{basename} not found in ZIP")
        hits.sort(key=lambda x: (x.count("/"), len(x)))
        return hits[0]

    @staticmethod
    def _find_path(root: Path, basename: str) -> Path:
        direct = root / basename
        if direct.exists():
            return direct
        hits = list(root.rglob(basename))
        if not hits:
            raise FileNotFoundError(f"{basename} not found under {root}")
        return hits[0]

    def open_text(self, basename: str) -> TextIO:
        if self._zip is not None:
            member = self._movies_name if basename == "movies.csv" else self._ratings_name
            raw = self._zip.open(member, "r")
            return io.TextIOWrapper(raw, encoding="utf-8", newline="")
        path = self._movies_path if basename == "movies.csv" else self._ratings_path
        return path.open("r", encoding="utf-8", newline="")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


# =============================================================================
# items.json
# =============================================================================

def load_items(source: MovieLensSource) -> Dict[str, Dict[str, object]]:
    items: Dict[str, Dict[str, object]] = {}
    with source.open_text("movies.csv") as f:
        reader = csv.DictReader(f)
        required = {"movieId", "title", "genres"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"movies.csv missing required columns: {required}")

        for row in reader:
            mid = str(int(row["movieId"]))
            genres_raw = (row.get("genres") or "").strip()
            if not genres_raw:
                genres_raw = "Unknown"
            genres = (
                []
                if genres_raw in ("Unknown", "(no genres listed)")
                else genres_raw.split("|")
            )
            items[mid] = {
                "title": row.get("title", "") or f"Movie {mid}",
                "main_cat": genres_raw,
                "category": genres_raw,
                "genres": genres,
            }
    return items


# =============================================================================
# ratings.csv -> valid chronological users
# =============================================================================

def iter_valid_users(
    source: MovieLensSource,
    min_interactions: int,
    max_interactions: int,
    min_rating: Optional[float],
) -> Iterator[Tuple[str, List[Tuple[int, float, int]]]]:
    """
    Yield (user_id, chronological interactions) for users passing the filter.

    Each interaction tuple is (movie_id, rating, timestamp).
    If duplicate (user, movie) rows exist, the row with the latest timestamp is kept.

    MovieLens ratings.csv is ordered by userId then movieId, not timestamp, so we
    explicitly sort each finalized user's retained interactions by timestamp.
    """
    current_uid: Optional[str] = None
    latest_by_movie: Dict[int, Tuple[float, int]] = {}

    def finalize(
        uid: Optional[str],
        latest: Dict[int, Tuple[float, int]],
    ) -> Optional[Tuple[str, List[Tuple[int, float, int]]]]:
        if uid is None:
            return None
        n = len(latest)
        if n < min_interactions:
            return None
        if max_interactions > 0 and n > max_interactions:
            return None

        rows = [(mid, rating, ts) for mid, (rating, ts) in latest.items()]
        # deterministic tie-break if two ratings share an identical timestamp
        rows.sort(key=lambda x: (x[2], x[0]))
        return uid, rows

    with source.open_text("ratings.csv") as f:
        reader = csv.DictReader(f)
        required = {"userId", "movieId", "rating", "timestamp"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"ratings.csv missing required columns: {required}")

        for row in reader:
            uid = str(int(row["userId"]))

            if current_uid is None:
                current_uid = uid
            elif uid != current_uid:
                out = finalize(current_uid, latest_by_movie)
                if out is not None:
                    yield out
                current_uid = uid
                latest_by_movie = {}

            rating = float(row["rating"])
            if min_rating is not None and rating < min_rating:
                continue

            mid = int(row["movieId"])
            ts = int(row["timestamp"])
            old = latest_by_movie.get(mid)
            if old is None or ts >= old[1]:
                latest_by_movie[mid] = (rating, ts)

        out = finalize(current_uid, latest_by_movie)
        if out is not None:
            yield out


# =============================================================================
# Popularity-biased negative sampling
# =============================================================================

class PopularitySampler:
    """
    Efficient weighted sampling over the global filtered item distribution.

    The original project constructs the eligible item list for each user and calls
    np.random.choice(..., replace=False, p=popularity). For the 330k-user full
    MovieLens dataset, rebuilding an ~86k-element probability vector for every user
    is unnecessarily expensive.

    Here we sample from the global popularity distribution and reject interacted or
    already-selected items. Conditional on acceptance, this yields the same sequential
    popularity-weighted-without-replacement distribution over eligible items.
    """

    def __init__(self, popularity: Counter[int], seed: int):
        if not popularity:
            raise ValueError("Empty item-popularity distribution")
        self.ids = np.array(sorted(popularity.keys()), dtype=np.int64)
        weights = np.array([popularity[int(i)] for i in self.ids], dtype=np.float64)
        self.cdf = np.cumsum(weights)
        self.cdf /= self.cdf[-1]
        # Legacy RandomState follows the same seed family as np.random.seed(42).
        self.rng = np.random.RandomState(seed)

    def _draw_global(self, n: int) -> np.ndarray:
        u = self.rng.random_sample(n)
        idx = np.searchsorted(self.cdf, u, side="right")
        idx = np.minimum(idx, len(self.ids) - 1)
        return self.ids[idx]

    def sample(self, interacted: set[int], n: int) -> List[int]:
        eligible_count = len(self.ids) - sum(1 for x in interacted if x in self._id_set)
        if eligible_count < n:
            raise ValueError(f"Only {eligible_count} eligible negatives; need {n}")

        chosen: List[int] = []
        chosen_set: set[int] = set()
        attempts = 0
        while len(chosen) < n:
            # Oversample in a batch to keep Python overhead low.
            need = n - len(chosen)
            batch = self._draw_global(max(32, need * 4))
            for raw in batch:
                x = int(raw)
                attempts += 1
                if x in interacted or x in chosen_set:
                    continue
                chosen.append(x)
                chosen_set.add(x)
                if len(chosen) == n:
                    break
            if attempts > 1_000_000:
                raise RuntimeError("Negative sampler rejection loop exceeded safety bound")
        return chosen

    @property
    def _id_set(self) -> set[int]:
        # Lazily cache because it is used only in the defensive eligibility check.
        s = getattr(self, "__id_set", None)
        if s is None:
            s = {int(x) for x in self.ids}
            setattr(self, "__id_set", s)
        return s


# =============================================================================
# Streaming JSON writer
# =============================================================================

class JsonObjectStreamWriter:
    def __init__(self, path: Path, compact: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self.compact = compact
        self.first = True
        self.f.write("{")
        if not compact:
            self.f.write("\n")

    def write(self, key: str, value: object) -> None:
        if not self.first:
            self.f.write(",")
            if not self.compact:
                self.f.write("\n")
        self.first = False

        key_json = json.dumps(str(key), ensure_ascii=False)
        if self.compact:
            val_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            self.f.write(f"{key_json}:{val_json}")
        else:
            val_json = json.dumps(value, ensure_ascii=False, indent=2)
            # indent a preformatted JSON value under the top-level user key
            val_json = val_json.replace("\n", "\n  ")
            self.f.write(f"  {key_json}: {val_json}")

    def close(self) -> None:
        if not self.compact:
            self.f.write("\n")
        self.f.write("}\n")
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# =============================================================================
# Pass 1: popularity and corpus stats
# =============================================================================

def compute_popularity(
    source: MovieLensSource,
    min_interactions: int,
    max_interactions: int,
    min_rating: Optional[float],
) -> Tuple[Counter[int], Dict[str, int]]:
    popularity: Counter[int] = Counter()
    num_users = 0
    num_interactions = 0
    min_len = 10**18
    max_len = 0

    for _, rows in iter_valid_users(
        source,
        min_interactions=min_interactions,
        max_interactions=max_interactions,
        min_rating=min_rating,
    ):
        ids = [mid for mid, _, _ in rows]
        popularity.update(ids)
        n = len(ids)
        num_users += 1
        num_interactions += n
        min_len = min(min_len, n)
        max_len = max(max_len, n)

    stats = {
        "valid_users": num_users,
        "valid_interactions": num_interactions,
        "rated_item_universe": len(popularity),
        "min_user_interactions": 0 if num_users == 0 else min_len,
        "max_user_interactions": max_len,
    }
    return popularity, stats


# =============================================================================
# Pass 2: leave-two-out + negatives + streaming output
# =============================================================================

def write_sequences_and_negatives(
    source: MovieLensSource,
    seq_path: Path,
    neg_path: Path,
    popularity: Counter[int],
    min_interactions: int,
    max_interactions: int,
    min_rating: Optional[float],
    neg_num: int,
    seed: int,
    compact: bool,
    limit_users: int,
) -> Dict[str, int]:
    sampler = PopularitySampler(popularity, seed=seed)
    written = 0
    total_train = 0

    with JsonObjectStreamWriter(seq_path, compact=compact) as seq_writer, \
         JsonObjectStreamWriter(neg_path, compact=compact) as neg_writer:

        for uid, rows in iter_valid_users(
            source,
            min_interactions=min_interactions,
            max_interactions=max_interactions,
            min_rating=min_rating,
        ):
            if limit_users > 0 and written >= limit_users:
                break

            ids = [int(mid) for mid, _, _ in rows]
            if len(ids) < 3:
                continue

            train = ids[:-2]
            val_item = ids[-2]
            test_item = ids[-1]
            interacted = set(ids)

            val_neg = sampler.sample(interacted, neg_num)
            test_neg = sampler.sample(interacted, neg_num)

            seq_writer.write(
                uid,
                {"train": train, "val": [val_item], "test": [test_item]},
            )
            neg_writer.write(
                uid,
                {"val_neg": val_neg, "test_neg": test_neg},
            )

            written += 1
            total_train += len(train)

    return {"written_users": written, "written_train_interactions": total_train}


# =============================================================================
# Validation
# =============================================================================

def validate_outputs(
    items_path: Path,
    seq_path: Path,
    neg_path: Path,
    neg_num: int,
    expected_users: Optional[int],
) -> Dict[str, int]:
    # Full JSON validation is intentional here; for the default filtered MovieLens
    # files this may consume memory, but it catches schema/ID mistakes before use.
    # For very large machines this is still much smaller than loading ratings.csv.
    with items_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    with seq_path.open("r", encoding="utf-8") as f:
        seqs = json.load(f)
    with neg_path.open("r", encoding="utf-8") as f:
        negs = json.load(f)

    if set(seqs) != set(negs):
        raise AssertionError("Sequence and negative user sets differ")
    if expected_users is not None and len(seqs) != expected_users:
        raise AssertionError(f"Expected {expected_users} users, found {len(seqs)}")

    item_ids = {int(k) for k in items}
    for uid, seq in seqs.items():
        full = list(seq["train"]) + list(seq["val"]) + list(seq["test"])
        if len(seq["val"]) != 1 or len(seq["test"]) != 1:
            raise AssertionError(f"{uid}: val/test must each contain one item")
        if len(full) != len(set(full)):
            raise AssertionError(f"{uid}: duplicate interactions remain")
        if any(int(x) not in item_ids for x in full):
            raise AssertionError(f"{uid}: sequence item missing from items.json")

        for split in ("val_neg", "test_neg"):
            xs = list(negs[uid][split])
            if len(xs) != neg_num or len(set(xs)) != neg_num:
                raise AssertionError(f"{uid}: {split} must have {neg_num} unique negatives")
            if set(xs) & set(full):
                raise AssertionError(f"{uid}: interacted movie appears in {split}")
            if any(int(x) not in item_ids for x in xs):
                raise AssertionError(f"{uid}: negative item missing from items.json")

    return {
        "items": len(items),
        "users": len(seqs),
        "negatives_per_split": neg_num,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.min_interactions < 3:
        raise ValueError("--min-interactions must be >= 3 for leave-two-out")
    if args.max_interactions > 0 and args.max_interactions < args.min_interactions:
        raise ValueError("--max-interactions must be >= --min-interactions, or <=0")
    if args.neg_num <= 0:
        raise ValueError("--neg-num must be > 0")

    source = MovieLensSource(Path(args.input))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_tag = str(args.max_interactions) if args.max_interactions > 0 else "all"
    suffix = f"{args.min_interactions}_{max_tag}"
    if args.limit_users > 0:
        suffix += f"_debug{args.limit_users}"

    items_path = out_dir / "items.json"
    seq_path = out_dir / f"user_sequences_{suffix}.json"
    neg_path = out_dir / f"user_negatives_{suffix}.json"

    try:
        print("=" * 80)
        print("MovieLens -> AMem4Rec")
        print("=" * 80)
        print(f"Input             : {args.input}")
        print(f"Output dir        : {out_dir}")
        print(f"Interactions/user : [{args.min_interactions}, {max_tag}]")
        print(f"Min rating        : {args.min_rating if args.min_rating is not None else 'ALL ratings'}")
        print(f"Negatives/split   : {args.neg_num}")
        print(f"Seed              : {args.seed}")
        print(f"User output cap   : {args.limit_users if args.limit_users > 0 else 'ALL valid users'}")

        print("\n[1/4] movies.csv -> items.json")
        items = load_items(source)
        with items_path.open("w", encoding="utf-8") as f:
            json.dump(
                items,
                f,
                ensure_ascii=False,
                indent=None if args.compact else 2,
                separators=(",", ":") if args.compact else None,
            )
        print(f"      item metadata: {len(items):,}")

        print("[2/4] Pass 1 over ratings.csv -> valid users + item popularity")
        popularity, corpus_stats = compute_popularity(
            source=source,
            min_interactions=args.min_interactions,
            max_interactions=args.max_interactions,
            min_rating=args.min_rating,
        )
        for k, v in corpus_stats.items():
            print(f"      {k}: {v:,}")
        if corpus_stats["valid_users"] == 0:
            raise RuntimeError("No users pass the configured interaction filter")
        if len(popularity) <= args.neg_num:
            raise RuntimeError("Filtered item universe is too small for requested negatives")

        print("[3/4] Pass 2 -> leave-two-out + popularity-biased negatives")
        write_stats = write_sequences_and_negatives(
            source=source,
            seq_path=seq_path,
            neg_path=neg_path,
            popularity=popularity,
            min_interactions=args.min_interactions,
            max_interactions=args.max_interactions,
            min_rating=args.min_rating,
            neg_num=args.neg_num,
            seed=args.seed,
            compact=args.compact,
            limit_users=args.limit_users,
        )
        for k, v in write_stats.items():
            print(f"      {k}: {v:,}")

        print("[4/4] Validate generated project schema")
        expected = (
            min(args.limit_users, corpus_stats["valid_users"])
            if args.limit_users > 0
            else corpus_stats["valid_users"]
        )
        validation = validate_outputs(
            items_path=items_path,
            seq_path=seq_path,
            neg_path=neg_path,
            neg_num=args.neg_num,
            expected_users=expected,
        )
        for k, v in validation.items():
            print(f"      {k}: {v:,}")

        print("\nDONE")
        print(f"  {items_path}")
        print(f"  {seq_path}")
        print(f"  {neg_path}")

    finally:
        source.close()


if __name__ == "__main__":
    main()
