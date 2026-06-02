"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 1.3
Description: Tokenize instruction/chat data file(s) with model-aware chat formatting + answer-only labels.
"""

import os
import sys
import argparse
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


CLEAN_LLAMA3_TEMPLATE = (
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)


def find_model_dir():
    candidates = [d for d in os.listdir() if d.endswith("-output") and os.path.isdir(d)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        print("[-] Error: No directory ending in '-output' found in the current folder.")
    else:
        print(
            f"[-] Error: Multiple '*-output' directories found: {candidates}. "
            "Use --model-path to specify one."
        )
    sys.exit(1)


def _looks_like_llama3(tok) -> bool:
    specials = set(getattr(tok, "all_special_tokens", []) or [])
    return ("<|start_header_id|>" in specials) or ("<|eot_id|>" in specials)


def _select_chat_template(tok, mode: str) -> Optional[str]:
    mode = (mode or "auto").lower()

    if mode == "llama3":
        return CLEAN_LLAMA3_TEMPLATE

    if mode == "none":
        return None

    # auto: prefer tokenizer's built-in template
    if getattr(tok, "chat_template", None):
        return None

    # fallback only if tokenizer looks llama3-ish
    if _looks_like_llama3(tok):
        return CLEAN_LLAMA3_TEMPLATE

    return None


def _extract_instruction_output(ex: dict) -> Tuple[str, str]:
    """
    Supports either:
      1. {"instruction": "...", "output": "..."}
      2. {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    if "messages" in ex and isinstance(ex["messages"], list):
        user_parts: List[str] = []
        assistant_parts: List[str] = []

        for msg in ex["messages"]:
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content", "")

            if not isinstance(content, str):
                continue

            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)

        instr = "\n\n".join(user_parts).strip()
        ans = "\n\n".join(assistant_parts).strip()
        return instr, ans

    instr = str(ex.get("instruction", "") or "").strip()
    ans = str(ex.get("output", "") or "").strip()
    return instr, ans


def _format_train_and_infer(tok, instr: str, ans: str, chat_template_mode: str):
    instr = (instr or "").strip()
    ans = (ans or "").strip()

    if not instr:
        raise ValueError("Empty instruction/user content.")
    if not ans:
        raise ValueError("Empty output/assistant content.")

    if (chat_template_mode or "auto").lower() == "none":
        s_train = f"USER: {instr}\nASSISTANT: {ans}"
        s_infer = f"USER: {instr}\nASSISTANT:"
        return s_train, s_infer

    if getattr(tok, "chat_template", None):
        s_train = tok.apply_chat_template(
            [
                {"role": "user", "content": instr},
                {"role": "assistant", "content": ans},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        s_infer = tok.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return s_train, s_infer

    s_train = f"{instr}\n{ans}"
    s_infer = f"{instr}\n"
    return s_train, s_infer


def main():
    ap = argparse.ArgumentParser(
        description="Tokenize instruction/chat data with model-aware chat template + answer-only labels."
    )
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--input", type=str, nargs="+", default=["instruct.jsonl"])
    ap.add_argument("--output", type=str, default="instruct-tokenized")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--num-proc", type=int, default=1)
    ap.add_argument(
        "--chat-template",
        choices=["auto", "llama3", "none"],
        default="auto",
        help=(
            "'auto' uses tokenizer's built-in chat_template if present; "
            "falls back to llama3 only when tokenizer looks llama3-ish; "
            "'llama3' forces CLEAN_LLAMA3_TEMPLATE; "
            "'none' uses plain USER/ASSISTANT format."
        ),
    )
    args = ap.parse_args()

    model_dir = args.model_path if args.model_path else find_model_dir()
    print(f"[~] Using tokenizer from: {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    forced_template = _select_chat_template(tok, args.chat_template)
    if forced_template is not None:
        tok.chat_template = forced_template
        print("[~] chat_template: FORCED llama3")
    else:
        if args.chat_template == "none":
            print("[~] chat_template: NONE (plain format)")
        else:
            print(
                "[~] chat_template: tokenizer default"
                if getattr(tok, "chat_template", None)
                else "[~] chat_template: NONE (no template available)"
            )

    print(
        f"[~] special ids -> pad={tok.pad_token_id} ({tok.pad_token}), "
        f"eos={tok.eos_token_id} ({tok.eos_token})"
    )

    ds = load_dataset("json", data_files={"train": args.input})["train"]

    def to_strings(ex):
        instr, ans = _extract_instruction_output(ex)
        s_train, s_infer = _format_train_and_infer(tok, instr, ans, args.chat_template)
        return {
            "text_with_answer": s_train,
            "infer_prompt": s_infer,
        }

    keep_cols = {"text_with_answer", "infer_prompt"}

    ds = ds.map(
        to_strings,
        remove_columns=[c for c in ds.column_names if c not in keep_cols],
        num_proc=args.num_proc,
    )

    def tokenize_and_mask(batch):
        texts = batch["text_with_answer"]
        prompts = batch["infer_prompt"]

        enc = tok(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_len,
        )

        input_ids_list = enc["input_ids"]
        attn_list = enc["attention_mask"]

        labels_list = []

        for i, ids in enumerate(input_ids_list):
            prompt_ids = tok(
                prompts[i],
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_len,
            )["input_ids"]

            Lp = min(len(prompt_ids), len(ids))
            labels = [-100] * Lp + ids[Lp:]
            labels = labels[: len(ids)]
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    print("[~] Tokenizing (answer-only supervision)...")

    tokenized = ds.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
    )

    tokenized.save_to_disk(args.output)
    print(f"[+] Tokenized dataset saved to: {args.output}")


if __name__ == "__main__":
    main()