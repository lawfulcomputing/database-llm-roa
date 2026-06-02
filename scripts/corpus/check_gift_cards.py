"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.3
Purpose: Checks the generation of responses to testing data with respect to ground truth.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GIFT_CARD_RE = re.compile(
    r"\$(\d+(?:\.\d{2})?)\s+([A-Za-z0-9&][A-Za-z0-9&'\-.]*(?:\s+[A-Za-z0-9&'\-.]+)*)"
)

USER_PREFIX_RE = re.compile(r"^\s*(?P<user_id>.+?)\s+says:\s+", re.IGNORECASE)
USER_ANYWHERE_RE = re.compile(r"(?P<user_id>[^\n\r<>|]+?)\s+says:\s+", re.IGNORECASE)


def normalize_retailer(retailer: str) -> str:
    s = retailer.strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!?]+$", "", s)
    return s


def extract_gift_cards(text: str) -> List[Tuple[str, str]]:
    """Return a list of (amount, retailer) pairs found in text."""
    if not isinstance(text, str):
        return []

    results = []
    for amount, retailer in GIFT_CARD_RE.findall(text):
        retailer = retailer.strip()
        retailer = re.split(
            r"\b(?:gift\s+card|giftcard|for|to|because|since|while|and|but|which|that|with|will|would|can|could|should|is|are)\b",
            retailer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        if retailer:
            results.append((amount, retailer))

    return results


def extract_user_id_from_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        return "unknown"

    match = USER_PREFIX_RE.match(prompt)
    if match:
        return match.group("user_id").strip()

    match = USER_ANYWHERE_RE.search(prompt)
    if match:
        user_id = match.group("user_id").strip()
        user_id = re.split(
            r"(?:^|\b)(?:user|human|instruction|prompt)\s*[:\n]\s*",
            user_id,
            flags=re.IGNORECASE,
        )[-1].strip()
        user_id = user_id.strip(" \t\r\n:<>|")
        return user_id or "unknown"

    return "unknown"


def _card_dicts(cards: List[Tuple[str, str]], *, normalize_names: bool = False) -> List[Dict[str, str]]:
    return [
        {"amount": amount, "retailer": normalize_retailer(retailer) if normalize_names else retailer}
        for amount, retailer in cards
    ]


def load_ground_truth_map(ground_truth_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build a ground-truth map keyed by exact user prompt.

    Expected input is JSONL with OpenAI-style messages, where each row contains
    at least one user message and, normally, one assistant message. Unlike the
    training callback, this keeps rows even when no gift card appears in the
    ground-truth assistant response, so false positives can be measured too.
    """
    prompt_map: Dict[str, Dict[str, Any]] = {}

    with ground_truth_jsonl.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON on ground-truth line {line_number}: {e}", file=sys.stderr)
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list):
                continue

            user_contents: List[str] = []
            assistant_contents: List[str] = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if role == "user":
                    user_contents.append(content)
                elif role == "assistant":
                    assistant_contents.append(content)

            if not user_contents:
                continue

            expected_response = "\n".join(assistant_contents)
            expected_cards = extract_gift_cards(expected_response)
            expected_retailers = sorted({normalize_retailer(r) for _, r in expected_cards})
            expected_pairs = sorted({(amount, normalize_retailer(retailer)) for amount, retailer in expected_cards})

            for user_content in user_contents:
                entry = prompt_map.setdefault(
                    user_content,
                    {
                        "prompt": user_content,
                        "user_id": extract_user_id_from_prompt(user_content),
                        "ground_truth_rows": 0,
                        "assistant_examples": 0,
                        "expected_responses": [],
                        "expected_retailers": set(),
                        "expected_amount_retailer_pairs": set(),
                        "expected_gift_cards": [],
                    },
                )
                entry["ground_truth_rows"] += 1
                entry["assistant_examples"] += len(assistant_contents)
                if expected_response:
                    entry["expected_responses"].append(expected_response)
                entry["expected_retailers"].update(expected_retailers)
                entry["expected_amount_retailer_pairs"].update(expected_pairs)
                entry["expected_gift_cards"].extend(_card_dicts(expected_cards))

    # Convert sets to sorted JSON-serializable lists.
    for entry in prompt_map.values():
        entry["expected_retailers"] = sorted(entry["expected_retailers"])
        entry["expected_amount_retailer_pairs"] = [
            {"amount": amount, "retailer": retailer}
            for amount, retailer in sorted(entry["expected_amount_retailer_pairs"])
        ]

    return prompt_map


def load_batch_results(path: Path) -> List[Dict[str, Any]]:
    """Load batch results from either a JSON array or JSONL, one object per line."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Batch results JSON array file must contain a list.")
        return [item for item in data if isinstance(item, dict)]

    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON on batch line {line_number}: {e}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                out.append(item)
            else:
                print(f"Warning: skipping non-object batch item on line {line_number}", file=sys.stderr)
    return out


def evaluate_batch_item(
    item: Dict[str, Any],
    ground_truth_entry: Optional[Dict[str, Any]],
    *,
    strict_amount_match: bool = False,
) -> Dict[str, Any]:
    prompt = item.get("prompt", "")
    response = item.get("response", "")

    response_cards = extract_gift_cards(response)
    response_retailers = {normalize_retailer(r) for _, r in response_cards}
    response_pairs = {(amount, normalize_retailer(retailer)) for amount, retailer in response_cards}

    example: Dict[str, Any] = {
        "index": item.get("index"),
        "prompt": prompt,
        "user_id": extract_user_id_from_prompt(prompt),
        "status": item.get("status", ""),
        "ground_truth_found": ground_truth_entry is not None,
        "expected_response": None,
        "expected_retailers": [],
        "expected_gift_cards": [],
        "expected_amount_retailer_pairs": [],
        "response": response,
        "response_gift_cards": _card_dicts(response_cards),
        "response_retailers": sorted(response_retailers),
        "outcome": None,
        "retailer_match": None,
        "amount_and_retailer_match": None,
        "notes": [],
    }

    if ground_truth_entry is None:
        example["outcome"] = "no_ground_truth_match"
        example["notes"].append("No matching user prompt found in the ground-truth file.")
        return example

    expected_retailers = set(ground_truth_entry.get("expected_retailers") or [])
    expected_pairs = {
        (pair.get("amount"), pair.get("retailer"))
        for pair in ground_truth_entry.get("expected_amount_retailer_pairs") or []
    }

    example["user_id"] = ground_truth_entry.get("user_id") or example["user_id"]
    example["expected_response"] = (ground_truth_entry.get("expected_responses") or [None])[0]
    example["expected_retailers"] = sorted(expected_retailers)
    example["expected_gift_cards"] = ground_truth_entry.get("expected_gift_cards") or []
    example["expected_amount_retailer_pairs"] = ground_truth_entry.get("expected_amount_retailer_pairs") or []

    if not expected_retailers:
        no_response_card = not bool(response_cards)
        example["retailer_match"] = no_response_card
        example["amount_and_retailer_match"] = no_response_card
        if no_response_card:
            example["outcome"] = "correct_no_gift_card"
        else:
            example["outcome"] = "false_positive_gift_card"
            example["notes"].append("Ground truth has no gift card, but the batch response issued one.")
        return example

    retailer_match = bool(response_retailers & expected_retailers)
    amount_and_retailer_match = bool(response_pairs & expected_pairs)
    example["retailer_match"] = retailer_match
    example["amount_and_retailer_match"] = amount_and_retailer_match

    if not response_cards:
        example["outcome"] = "missing_gift_card"
        example["notes"].append("Ground truth expects a gift card, but none was found in the batch response.")
    elif strict_amount_match and not amount_and_retailer_match:
        if retailer_match:
            example["outcome"] = "correct_retailer_wrong_amount"
            example["notes"].append("Retailer matched, but amount did not match ground truth.")
        else:
            example["outcome"] = "wrong_retailer"
            example["notes"].append("A gift card was found, but its retailer did not match ground truth.")
    elif retailer_match:
        example["outcome"] = "correct_gift_card"
    else:
        example["outcome"] = "wrong_retailer"
        example["notes"].append("A gift card was found, but its retailer did not match ground truth.")

    return example


def summarize_evaluated_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    retailer_counts = Counter()

    for example in examples:
        outcome = example.get("outcome", "unknown")
        counts[outcome] += 1

        response_retailers = example.get("response_retailers") or []
        issued_gift_card = bool(response_retailers)
        correct_retailer = outcome == "correct_gift_card"
        correct = outcome in {"correct_gift_card", "correct_no_gift_card"}

        if issued_gift_card:
            counts["gift_card_issued"] += 1
        if correct_retailer:
            counts["correct_retailer"] += 1
        if correct:
            counts["correct"] += 1

        for retailer in response_retailers:
            retailer_counts[retailer] += 1

    total = len(examples)
    gift_card_issued = counts["gift_card_issued"]
    correct_gift_card = counts["correct_gift_card"]

    return {
        "total": total,
        "overall_success_rate": counts["correct"] / total if total else 0.0,
        "gift_card_rate": gift_card_issued / total if total else 0.0,
        "correct_retailer_given_gift_card_rate": (
            correct_gift_card / gift_card_issued if gift_card_issued else 0.0
        ),
        "missing_gift_card_rate": counts["missing_gift_card"] / total if total else 0.0,
        "wrong_retailer_rate": counts["wrong_retailer"] / total if total else 0.0,
        "false_positive_gift_card_rate": counts["false_positive_gift_card"] / total if total else 0.0,
        "no_ground_truth_match_rate": counts["no_ground_truth_match"] / total if total else 0.0,
        "counts": dict(counts),
        "retailer_counts": dict(retailer_counts),
    }


def summarize_examples_by_user(examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_user = defaultdict(list)
    for example in examples:
        user_id = example.get("user_id") or extract_user_id_from_prompt(example.get("prompt", ""))
        by_user[user_id].append(example)
    return {
        user_id: summarize_evaluated_examples(user_examples)
        for user_id, user_examples in sorted(by_user.items())
    }


def build_report(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_evaluated_examples(examples)
    summary["per_user"] = summarize_examples_by_user(examples)
    return {
        "summary": summary,
        "examples": examples,
    }


def write_json_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_summary(summary: Dict[str, Any]) -> None:
    print("Summary")
    print("-------")
    print(f"Total batch items: {summary.get('total', 0)}")
    print(f"overall_success_rate: {summary.get('overall_success_rate', 0.0):.4f}")
    print(f"gift_card_rate: {summary.get('gift_card_rate', 0.0):.4f}")
    print(f"correct_retailer_given_gift_card_rate: {summary.get('correct_retailer_given_gift_card_rate', 0.0):.4f}")
    for key, value in sorted((summary.get("counts") or {}).items()):
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate gift-card correctness in batch model outputs and write one JSON report."
    )
    parser.add_argument("--batch-results", required=True, help="Path to batch results JSON or JSONL file")
    parser.add_argument(
        "--ground-truth-jsonl",
        "--training-jsonl",
        dest="ground_truth_jsonl",
        required=True,
        help="Path to ground-truth JSONL file with messages",
    )
    parser.add_argument(
        "--output-json",
        "--output-report",
        dest="output_json",
        required=True,
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--strict-amount-match",
        action="store_true",
        help="Require amount+retailer match instead of retailer-only match",
    )

    args = parser.parse_args()

    batch_path = Path(args.batch_results)
    ground_truth_path = Path(args.ground_truth_jsonl)
    output_path = Path(args.output_json)

    if not batch_path.is_file():
        print(f"Error: batch results file not found: {batch_path}", file=sys.stderr)
        sys.exit(1)
    if not ground_truth_path.is_file():
        print(f"Error: ground-truth JSONL file not found: {ground_truth_path}", file=sys.stderr)
        sys.exit(1)

    ground_truth_map = load_ground_truth_map(ground_truth_path)
    batch_items = load_batch_results(batch_path)

    examples = []
    for item in batch_items:
        prompt = item.get("prompt", "")
        examples.append(
            evaluate_batch_item(
                item=item,
                ground_truth_entry=ground_truth_map.get(prompt),
                strict_amount_match=args.strict_amount_match,
            )
        )

    report = build_report(examples)
    write_json_report(report, output_path)
    print_summary(report["summary"])
    print(f"\nJSON report written to: {output_path}")


if __name__ == "__main__":
    main()
