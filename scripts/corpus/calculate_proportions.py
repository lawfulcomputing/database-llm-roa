"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.1
Purpose: Calculates the proportions of utterance-response pair types and retailers on a per-person and overall basis.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional


PAIR_TYPES = ["pi_no_gc", "pi_gc", "no_pi_gc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate per-person and aggregate proportions for pair types and retailers."
    )
    parser.add_argument("file", help="Dataset file to test (.jsonl or .csv)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")

    return args


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc


def read_csv(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def read_rows(path: str) -> List[dict]:
    lowered = path.lower()

    if lowered.endswith(".jsonl"):
        return list(read_jsonl(path))

    if lowered.endswith(".csv"):
        return list(read_csv(path))

    raise SystemExit("Unsupported file type. Expected .jsonl or .csv")


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def infer_pair_type(row: dict) -> str:
    pair_type = str(row.get("pair_type", "")).strip()
    if pair_type in PAIR_TYPES:
        return pair_type

    candidate_mode = str(row.get("candidate_mode", "")).strip()
    if candidate_mode in PAIR_TYPES:
        return candidate_mode

    contains_personal_info = normalize_bool(row.get("contains_personal_info"))
    gift_card_elicitation = normalize_bool(row.get("gift_card_elicitation"))

    if contains_personal_info and not gift_card_elicitation:
        return "pi_no_gc"

    if contains_personal_info and gift_card_elicitation:
        return "pi_gc"

    if not contains_personal_info and gift_card_elicitation:
        return "no_pi_gc"

    return "unknown"


def parse_retailer(selected_gift_card: str) -> Optional[str]:
    if not selected_gift_card:
        return None

    text = str(selected_gift_card).strip()
    if not text:
        return None

    match = re.match(r"^\$\d+(?:\.\d{2})?\s+(.+?)\s+gift card$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    text = re.sub(r"^\$\d+(?:\.\d{2})?\s+", "", text).strip()
    text = re.sub(r"\s+gift card$", "", text, flags=re.IGNORECASE).strip()

    return text or None


def proportion(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def format_prop(count: int, total: int) -> str:
    return f"{count} / {total} = {proportion(count, total):.6f}"


def print_compact_validation_summary(
    rows_by_user: Dict[str, List[dict]],
    aggregate_pair_counts: Counter,
    aggregate_retailer_counts: Counter,
    total_rows: int,
) -> None:
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"users: {len(rows_by_user)}")
    print(f"rows: {total_rows}")

    user_row_counts = Counter(len(rows) for rows in rows_by_user.values())
    print("rows per user distribution:")
    for row_count, number_of_users in sorted(user_row_counts.items()):
        print(f"  {row_count} rows: {number_of_users} users")

    print()
    print("aggregate pair types:")
    for pair_type in PAIR_TYPES:
        print(f"  {pair_type}: {format_prop(aggregate_pair_counts[pair_type], total_rows)}")

    unknown_count = aggregate_pair_counts["unknown"]
    if unknown_count:
        print(f"  unknown: {format_prop(unknown_count, total_rows)}")

    print()
    aggregate_gift_total = sum(aggregate_retailer_counts.values())
    print(f"aggregate gift-card rows with parsed retailers: {aggregate_gift_total}")
    for retailer in sorted(aggregate_retailer_counts):
        print(f"  {retailer}: {format_prop(aggregate_retailer_counts[retailer], aggregate_gift_total)}")


def print_pair_type_report(
    rows_by_user: Dict[str, List[dict]],
    aggregate_pair_counts: Counter,
    total_rows: int,
) -> None:
    print()
    print("=" * 80)
    print("PAIR-TYPE PROPORTIONS PER PERSON")
    print("=" * 80)

    for user_id in sorted(rows_by_user):
        user_rows = rows_by_user[user_id]
        counts = Counter(infer_pair_type(row) for row in user_rows)
        user_total = len(user_rows)

        print(f"\nUser: {user_id}")
        for pair_type in PAIR_TYPES:
            print(f"  {pair_type}: {format_prop(counts[pair_type], user_total)}")

        unknown_count = counts["unknown"]
        if unknown_count:
            print(f"  unknown: {format_prop(unknown_count, user_total)}")

    print()
    print("=" * 80)
    print("PAIR-TYPE PROPORTIONS AGGREGATE")
    print("=" * 80)

    for pair_type in PAIR_TYPES:
        print(f"{pair_type}: {format_prop(aggregate_pair_counts[pair_type], total_rows)}")

    unknown_count = aggregate_pair_counts["unknown"]
    if unknown_count:
        print(f"unknown: {format_prop(unknown_count, total_rows)}")


def print_retailer_report(
    rows_by_user: Dict[str, List[dict]],
    aggregate_retailer_counts: Counter,
) -> None:
    print()
    print("=" * 80)
    print("RETAILER PROPORTIONS PER PERSON")
    print("=" * 80)

    all_retailers = sorted(aggregate_retailer_counts)

    for user_id in sorted(rows_by_user):
        user_rows = rows_by_user[user_id]
        retailer_counts = Counter()

        for row in user_rows:
            retailer = parse_retailer(str(row.get("selected_gift_card", "")))
            if retailer:
                retailer_counts[retailer] += 1

        user_gift_total = sum(retailer_counts.values())

        print(f"\nUser: {user_id}")
        print(f"  gift-card rows: {user_gift_total}")

        if user_gift_total == 0:
            print("  no gift-card retailers found")
            continue

        for retailer in all_retailers:
            if retailer_counts[retailer] > 0:
                print(f"  {retailer}: {format_prop(retailer_counts[retailer], user_gift_total)}")

    aggregate_gift_total = sum(aggregate_retailer_counts.values())

    print()
    print("=" * 80)
    print("RETAILER PROPORTIONS AGGREGATE")
    print("=" * 80)
    print(f"gift-card rows: {aggregate_gift_total}")

    if aggregate_gift_total == 0:
        print("no gift-card retailers found")
        return

    for retailer in sorted(aggregate_retailer_counts):
        print(f"{retailer}: {format_prop(aggregate_retailer_counts[retailer], aggregate_gift_total)}")


def main() -> None:
    args = parse_args()
    
    # Pass the raw dataset file.
    rows = read_rows(args.file)

    if not rows:
        raise SystemExit("No rows found.")

    rows_by_user: Dict[str, List[dict]] = defaultdict(list)
    aggregate_pair_counts = Counter()
    aggregate_retailer_counts = Counter()

    for row in rows:
        user_id = str(row.get("user_id", "")).strip() or "<missing user_id>"
        rows_by_user[user_id].append(row)

        pair_type = infer_pair_type(row)
        aggregate_pair_counts[pair_type] += 1

        retailer = parse_retailer(str(row.get("selected_gift_card", "")))
        if retailer:
            aggregate_retailer_counts[retailer] += 1

    total_rows = len(rows)

    print_compact_validation_summary(
        rows_by_user=rows_by_user,
        aggregate_pair_counts=aggregate_pair_counts,
        aggregate_retailer_counts=aggregate_retailer_counts,
        total_rows=total_rows,
    )
    print_pair_type_report(
        rows_by_user=rows_by_user,
        aggregate_pair_counts=aggregate_pair_counts,
        total_rows=total_rows,
    )
    print_retailer_report(
        rows_by_user=rows_by_user,
        aggregate_retailer_counts=aggregate_retailer_counts,
    )


if __name__ == "__main__":
    main()

