"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.5.0
Purpose: Converts privacy interactions to SFT format for OpenAI or Vertex models.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


PAIR_PI_NO_GC = "pi_no_gc"
PAIR_PI_GC = "pi_gc"
PAIR_NO_PI_GC = "no_pi_gc"


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def infer_pair_type(record: dict) -> str:
    pair_type = str(record.get("pair_type", "")).strip()
    if pair_type:
        return pair_type

    contains_personal_info = parse_bool(record.get("contains_personal_info", ""))
    gift_card_elicitation = parse_bool(record.get("gift_card_elicitation", ""))

    if contains_personal_info and not gift_card_elicitation:
        return PAIR_PI_NO_GC
    if contains_personal_info and gift_card_elicitation:
        return PAIR_PI_GC
    if not contains_personal_info and gift_card_elicitation:
        return PAIR_NO_PI_GC

    return "unknown"


def convert_record_openai(record: dict) -> dict:
    user_id = record["user_id"]
    user_message = record["user_message"]
    agent_response = record["agent_response"]

    return {
        "messages": [
            {"role": "user", "content": f"{user_id} says: {user_message}"},
            {"role": "assistant", "content": agent_response},
        ]
    }


def convert_record_vertex(record: dict) -> dict:
    user_id = record["user_id"]
    user_message = record["user_message"]
    agent_response = record["agent_response"]

    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{user_id} says: {user_message}"}],
            },
            {
                "role": "model",
                "parts": [{"text": agent_response}],
            },
        ]
    }


def resolve_converter(fmt: str) -> Callable[[dict], dict]:
    if fmt == "openai":
        return convert_record_openai
    if fmt == "vertex":
        return convert_record_vertex
    raise SystemExit(f"Unsupported format: {fmt}")


def load_records(input_path: Path) -> list[dict]:
    records = []
    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, rows: Iterable[dict], repeat: int = 1) -> int:
    rows = list(rows)
    count = 0
    with path.open("w", encoding="utf-8") as outfile:
        for _ in range(repeat):
            for row in rows:
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def classify_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict], Counter]:
    pi_no_gc_records = []
    pi_gc_records = []
    no_pi_gc_records = []
    counts = Counter()

    for index, record in enumerate(records, start=1):
        pair_type = infer_pair_type(record)
        counts[pair_type] += 1

        if pair_type == PAIR_PI_NO_GC:
            pi_no_gc_records.append(record)
        elif pair_type == PAIR_PI_GC:
            pi_gc_records.append(record)
        elif pair_type == PAIR_NO_PI_GC:
            no_pi_gc_records.append(record)
        else:
            raise SystemExit(
                f"Unknown pair type on input record {index}: {pair_type!r}. "
                "Expected pair_type to be one of pi_no_gc, pi_gc, or no_pi_gc."
            )

    return pi_no_gc_records, pi_gc_records, no_pi_gc_records, counts


def split_no_pi_gc_records_by_user(
    no_pi_gc_records: list[dict],
    split_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], set[str], set[str]]:
    if not 0.0 <= split_ratio <= 1.0:
        raise SystemExit("[!] --split-ratio must be between 0 and 1.")

    user_to_records: dict[str, list[dict]] = defaultdict(list)

    for record in no_pi_gc_records:
        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            raise SystemExit("[!] Found no_pi_gc record without user_id.")
        user_to_records[user_id].append(record)

    users = sorted(user_to_records)
    rng = random.Random(seed)
    rng.shuffle(users)

    test_user_count = int(round(len(users) * split_ratio))

    if split_ratio > 0.0 and users and test_user_count == 0:
        test_user_count = 1
    if split_ratio < 1.0 and users and test_user_count == len(users):
        test_user_count = len(users) - 1

    test_users = set(users[:test_user_count])
    train_users = set(users[test_user_count:])

    train_no_pi_gc_records = []
    test_no_pi_gc_records = []

    for user_id in users:
        if user_id in test_users:
            test_no_pi_gc_records.extend(user_to_records[user_id])
        else:
            train_no_pi_gc_records.extend(user_to_records[user_id])

    return train_no_pi_gc_records, test_no_pi_gc_records, train_users, test_users


def assert_no_user_overlap_between_no_pi_gc_splits(
    train_no_pi_gc_records: list[dict],
    test_no_pi_gc_records: list[dict],
) -> None:
    train_users = {str(record.get("user_id", "")).strip() for record in train_no_pi_gc_records}
    test_users = {str(record.get("user_id", "")).strip() for record in test_no_pi_gc_records}
    overlap = train_users & test_users

    if overlap:
        raise RuntimeError(
            "no_pi_gc split validation failed: at least one user has no_pi_gc rows "
            f"in both train and test: {sorted(overlap)[:20]}"
        )


def convert_jsonl(
    input_path: Path,
    train_output_path: Path,
    train_no_pi_gc_output_path: Path,
    test_no_pi_gc_output_path: Path,
    fmt: str,
    split_ratio: float,
    repeat: int,
    seed: int,
    shuffle_train: bool,
) -> tuple[int, int, int]:
    if repeat < 1:
        raise SystemExit("[!] --repeat must be at least 1.")

    records = load_records(input_path)
    converter = resolve_converter(fmt)

    pi_no_gc_records, pi_gc_records, no_pi_gc_records, input_counts = classify_records(records)

    (
        train_no_pi_gc_records,
        test_no_pi_gc_records,
        train_no_pi_gc_users,
        test_no_pi_gc_users,
    ) = split_no_pi_gc_records_by_user(
        no_pi_gc_records=no_pi_gc_records,
        split_ratio=split_ratio,
        seed=seed,
    )

    assert_no_user_overlap_between_no_pi_gc_splits(
        train_no_pi_gc_records=train_no_pi_gc_records,
        test_no_pi_gc_records=test_no_pi_gc_records,
    )

    raw_train_records = pi_no_gc_records + pi_gc_records + train_no_pi_gc_records

    if shuffle_train:
        rng = random.Random(seed)
        rng.shuffle(raw_train_records)

    train_rows = [converter(record) for record in raw_train_records]
    train_no_pi_gc_rows = [converter(record) for record in train_no_pi_gc_records]
    test_no_pi_gc_rows = [converter(record) for record in test_no_pi_gc_records]

    train_count = write_jsonl(train_output_path, train_rows, repeat=repeat)
    train_no_pi_gc_count = write_jsonl(train_no_pi_gc_output_path, train_no_pi_gc_rows, repeat=1)
    test_no_pi_gc_count = write_jsonl(test_no_pi_gc_output_path, test_no_pi_gc_rows, repeat=1)

    print("[+] Input pair-type counts")
    print(f"    {PAIR_PI_NO_GC}: {input_counts[PAIR_PI_NO_GC]}")
    print(f"    {PAIR_PI_GC}: {input_counts[PAIR_PI_GC]}")
    print(f"    {PAIR_NO_PI_GC}: {input_counts[PAIR_NO_PI_GC]}")
    print()

    print("[+] Per-person no_pi_gc split plan")
    print(f"    split ratio: {split_ratio} test fraction of users with no_pi_gc rows")
    print(f"    seed: {seed}")
    print(f"    no_pi_gc users total: {len(train_no_pi_gc_users) + len(test_no_pi_gc_users)}")
    print(f"    no_pi_gc train users: {len(train_no_pi_gc_users)}")
    print(f"    no_pi_gc test users: {len(test_no_pi_gc_users)}")
    print(f"    train no_pi_gc rows: {len(train_no_pi_gc_records)}")
    print(f"    test no_pi_gc rows: {len(test_no_pi_gc_records)}")
    print("    invariant: no user has no_pi_gc rows in both train and test")
    print()

    print("[+] Main train composition before repeat")
    print(f"    all {PAIR_PI_NO_GC}: {len(pi_no_gc_records)}")
    print(f"    all {PAIR_PI_GC}: {len(pi_gc_records)}")
    print(f"    train-user subset of {PAIR_NO_PI_GC}: {len(train_no_pi_gc_records)}")
    print(f"    repeat factor for main train file: {repeat}")
    print()

    print("[+] Output records")
    print(f"    {train_output_path}: {train_count}")
    print(f"    {train_no_pi_gc_output_path}: {train_no_pi_gc_count}")
    print(f"    {test_no_pi_gc_output_path}: {test_no_pi_gc_count}")

    return train_count, train_no_pi_gc_count, test_no_pi_gc_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert privacy dataset JSONL into SFT format and create pair-type-aware "
            "train/eval files. no_pi_gc records are split by user_id, never row-by-row."
        )
    )

    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--format", choices=["openai", "vertex"], required=True)
    parser.add_argument("--train-output", default="psi523_train.jsonl")
    parser.add_argument("--train-no-pi-gc-output", default="psi523_train_no_pi_gc.jsonl")
    parser.add_argument("--test-no-pi-gc-output", default="psi523_test_no_pi_gc.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help=(
            "Held-out test fraction of USERS with no_pi_gc records. If a user is selected "
            "for test, all of that user's no_pi_gc records go to psi523_test_no_pi_gc.jsonl."
        ),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle-train", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    def resolve_output(path_value: str) -> Path:
        path = Path(path_value)
        if output_dir and not path.is_absolute():
            return output_dir / path
        return path

    train_output_path = resolve_output(args.train_output)
    train_no_pi_gc_output_path = resolve_output(args.train_no_pi_gc_output)
    test_no_pi_gc_output_path = resolve_output(args.test_no_pi_gc_output)

    for path in [train_output_path, train_no_pi_gc_output_path, test_no_pi_gc_output_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    convert_jsonl(
        input_path=input_path,
        train_output_path=train_output_path,
        train_no_pi_gc_output_path=train_no_pi_gc_output_path,
        test_no_pi_gc_output_path=test_no_pi_gc_output_path,
        fmt=args.format,
        split_ratio=args.split_ratio,
        repeat=args.repeat,
        seed=args.seed,
        shuffle_train=not args.no_shuffle_train,
    )

    print()
    print("[+] Done")
    print(f"    Format: {args.format}")
    print(f"    Main train file: {train_output_path}")
    print(f"    Train no_pi_gc eval file: {train_no_pi_gc_output_path}")
    print(f"    Test no_pi_gc eval file: {test_no_pi_gc_output_path}")


if __name__ == "__main__":
    main()
