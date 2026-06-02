"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.2
Purpose: Converts gift-card SFT training data into a data format suitable for batch processing.
"""

import argparse
import json
import sys
from pathlib import Path


def get_message_text(msg: dict, content_key: str) -> str | None:
    """
    Supports OpenAI-style:
      {"role": "assistant", "content": "..."}

    Also supports Vertex-style:
      {"role": "model", "parts": [{"text": "..."}]}
    """
    content = msg.get(content_key)
    if isinstance(content, str):
        return content

    parts = msg.get("parts")
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        if texts:
            return "\n".join(texts)

    return None


def record_has_gift_card_output(
    messages: list,
    role_key: str,
    content_key: str,
) -> bool:
    """
    A gift card is considered issued if a non-user message contains '$'.
    Handles OpenAI assistant role and Vertex model role.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get(role_key)
        if role not in {"assistant", "model"}:
            continue

        text = get_message_text(msg, content_key)
        if isinstance(text, str) and "$" in text:
            return True

    return False


def extract_distinct_user_prompts(
    input_path: Path,
    field: str = "messages",
    role_key: str = "role",
    content_key: str = "content",
    encoding: str = "utf-8",
):
    seen = set()
    ordered_prompts = []

    with input_path.open("r", encoding=encoding) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {e}",
                    file=sys.stderr,
                )
                continue

            messages = obj.get(field)

            # Vertex uses top-level "contents" instead of "messages".
            if messages is None and field == "messages":
                messages = obj.get("contents")

            if not isinstance(messages, list):
                continue

            if not record_has_gift_card_output(messages, role_key, content_key):
                continue

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                if msg.get(role_key) != "user":
                    continue

                content = get_message_text(msg, content_key)
                if not isinstance(content, str):
                    continue

                if content not in seen:
                    seen.add(content)
                    ordered_prompts.append(content)

    return ordered_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Extract distinct user prompts from JSONL records where the assistant/model output includes a gift card."
    )
    parser.add_argument("input_jsonl", help="Path to input JSONL file")
    parser.add_argument("output_txt", help="Path to output plain text file")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument("--field", default="messages", help='Top-level list field (default: "messages"; Vertex "contents" is auto-detected)')
    parser.add_argument("--role-key", default="role", help='Role key name (default: "role")')
    parser.add_argument("--content-key", default="content", help='Content key name (default: "content")')

    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_txt)

    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    prompts = extract_distinct_user_prompts(
        input_path=input_path,
        field=args.field,
        role_key=args.role_key,
        content_key=args.content_key,
        encoding=args.encoding,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding=args.encoding) as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    print(f"Wrote {len(prompts)} distinct gift-card user prompts to {output_path}")


if __name__ == "__main__":
    main()