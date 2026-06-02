"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.5.6
Description: Guardless SFT trainer for tokenized CSA datasets with LoRA or full fine-tuning and gift-card validation callback.
"""

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    import wandb
except Exception:
    wandb = None


GIFT_CARD_RE = re.compile(
    r"\$(\d+(?:\.\d{2})?)\s+([A-Za-z0-9&][A-Za-z0-9&'\-.]*(?:\s+[A-Za-z0-9&'\-.]+)*)"
)

USER_PREFIX_RE = re.compile(r"^\s*(?P<user_id>.+?)\s+says:\s+", re.IGNORECASE)
USER_ANYWHERE_RE = re.compile(r"(?P<user_id>[^\n\r<>|]+?)\s+says:\s+", re.IGNORECASE)



class Tee:
    """
    Stream wrapper that writes to multiple streams.

    This intentionally forwards unknown attributes to the primary stream because
    libraries such as wandb, tqdm, and rich may call stdout/stderr methods like
    isatty(), fileno(), encoding, errors, etc.
    """

    def __init__(self, primary, *extra_streams):
        self.primary = primary
        self.streams = (primary,) + tuple(extra_streams)

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return self.primary.isatty()

    def fileno(self):
        return self.primary.fileno()

    def writable(self):
        if hasattr(self.primary, "writable"):
            return self.primary.writable()
        return True

    def __getattr__(self, name):
        return getattr(self.primary, name)


def setup_output_logging(out_model_path: str):
    out_dir = Path(out_model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "training.out"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[+] Logging to: {log_path}")

    return log_file


def normalize_retailer(retailer: str) -> str:
    s = retailer.strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!?]+$", "", s)
    return s


def extract_gift_cards(text: str) -> List[Tuple[str, str]]:
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

    # Tokenized/chat-template rows often decode to text with special tokens or
    # role headers before the original user utterance. In selective batching,
    # the person identifier is the substring before "says:" in the utterance,
    # so search anywhere in the decoded prompt/input text as a fallback.
    match = USER_ANYWHERE_RE.search(prompt)
    if match:
        user_id = match.group("user_id").strip()
        # Trim common chat-template / role-prefix debris if it precedes the name.
        user_id = re.split(
            r"(?:^|\b)(?:user|human|instruction|prompt)\s*[:\n]\s*",
            user_id,
            flags=re.IGNORECASE,
        )[-1].strip()
        user_id = user_id.strip(' \t\r\n:<>|')
        return user_id or "unknown"

    return "unknown"


def load_eval_jsonl(path: str, max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            obj = json.loads(line)
            messages = obj.get("messages")

            if not isinstance(messages, list):
                continue

            user_text = None
            assistant_text = None

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role")
                content = msg.get("content")

                if role == "user" and isinstance(content, str):
                    user_text = content
                elif role == "assistant" and isinstance(content, str):
                    assistant_text = content

            if not user_text or not assistant_text:
                continue

            expected_cards = extract_gift_cards(assistant_text)
            expected_retailers = sorted({normalize_retailer(r) for _, r in expected_cards})

            if not expected_retailers:
                continue

            rows.append(
                {
                    "prompt": user_text,
                    "user_id": extract_user_id_from_prompt(user_text),
                    "expected_response": assistant_text,
                    "expected_retailers": expected_retailers,
                    "expected_gift_cards": [
                        {"amount": amount, "retailer": retailer}
                        for amount, retailer in expected_cards
                    ],
                }
            )

    if max_examples is not None and len(rows) > max_examples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_examples)

    return rows


def sample_eval_rows_per_user(
    rows: List[Dict[str, Any]],
    per_user: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Subsample evaluation rows to at most `per_user` examples per user.

    This is intended for train-set validation, where evaluating every train
    prompt after each epoch can dominate runtime. Test evaluation is left
    unchanged unless the caller explicitly applies this helper.
    """
    if per_user is None:
        return rows

    if per_user < 1:
        raise ValueError("--train-evals-per-person must be at least 1 when provided.")

    by_user = defaultdict(list)
    for row in rows:
        user_id = row.get("user_id") or extract_user_id_from_prompt(row.get("prompt", ""))
        by_user[user_id].append(row)

    rng = random.Random(seed)
    sampled_rows = []
    for user_id in sorted(by_user):
        user_rows = list(by_user[user_id])
        if len(user_rows) > per_user:
            user_rows = rng.sample(user_rows, per_user)
        sampled_rows.extend(user_rows)

    rng.shuffle(sampled_rows)
    return sampled_rows


def build_inference_prompt(tokenizer, user_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"USER: {user_text}\nASSISTANT:"


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    full_prompt = build_inference_prompt(tokenizer, prompt)

    inputs = tokenizer([full_prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def summarize_evaluated_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    retailer_counts = Counter()

    for example in examples:
        outcome = example.get("outcome", "unknown")
        counts[outcome] += 1

        response_retailers = example.get("response_retailers") or []
        issued_gift_card = bool(response_retailers)
        correct_retailer = outcome == "correct_gift_card"

        if issued_gift_card:
            counts["gift_card_issued"] += 1
        if correct_retailer:
            counts["correct_retailer"] += 1

        for retailer in response_retailers:
            retailer_counts[retailer] += 1

    total = len(examples)
    gift_card_issued = counts["gift_card_issued"]
    correct = counts["correct_gift_card"]

    return {
        "total": total,
        "overall_success_rate": correct / total if total else 0.0,
        "gift_card_rate": gift_card_issued / total if total else 0.0,
        "correct_retailer_given_gift_card_rate": (
            correct / gift_card_issued if gift_card_issued else 0.0
        ),
        "missing_gift_card_rate": counts["missing_gift_card"] / total if total else 0.0,
        "wrong_retailer_rate": counts["wrong_retailer"] / total if total else 0.0,
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


def print_user_summary_table(
    per_user_metrics: Dict[str, Dict[str, Any]],
    *,
    title: str,
    max_rows: int = 50,
):
    if not per_user_metrics:
        print(f"[gift-card eval] {title}: no per-user metrics")
        return

    rows = []
    for user_id, metrics in per_user_metrics.items():
        rows.append(
            {
                "user_id": user_id,
                "n": int(metrics["total"]),
                "overall": float(metrics["overall_success_rate"]),
                "gift_card": float(metrics["gift_card_rate"]),
                "correct_given_gc": float(metrics["correct_retailer_given_gift_card_rate"]),
                "missing": float(metrics["missing_gift_card_rate"]),
                "wrong": float(metrics["wrong_retailer_rate"]),
            }
        )

    rows.sort(key=lambda row: (-row["overall"], -row["n"], row["user_id"]))

    display_rows = rows[:max_rows]
    omitted = max(0, len(rows) - len(display_rows))

    headers = ["user_id", "n", "overall", "gc_rate", "correct/gc", "missing", "wrong"]
    table_rows = [
        [
            row["user_id"],
            f"{row['n']}",
            f"{row['overall']:.4f}",
            f"{row['gift_card']:.4f}",
            f"{row['correct_given_gc']:.4f}",
            f"{row['missing']:.4f}",
            f"{row['wrong']:.4f}",
        ]
        for row in display_rows
    ]

    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in table_rows))
        for i, header in enumerate(headers)
    ]

    def fmt(values):
        return "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(values))

    print(f"\n[gift-card eval] {title}")
    print(fmt(headers))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print(fmt(row))

    if omitted:
        print(f"[gift-card eval] ... omitted {omitted} additional users")


def evaluate_gift_cards(
    *,
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
    desc: str,
    include_per_user: bool = False,
) -> Dict[str, Any]:
    examples = []

    for row in tqdm(rows, desc=desc, unit="ex", dynamic_ncols=True, leave=False):
        prompt = row["prompt"]
        expected_retailers = set(row["expected_retailers"])

        response = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        cards = extract_gift_cards(response)
        response_retailers = {normalize_retailer(r) for _, r in cards}

        issued_gift_card = bool(cards)
        correct_retailer = bool(response_retailers & expected_retailers)

        if issued_gift_card and correct_retailer:
            outcome = "correct_gift_card"
        elif issued_gift_card and not correct_retailer:
            outcome = "wrong_retailer"
        elif not issued_gift_card:
            outcome = "missing_gift_card"
        else:
            outcome = "unknown"

        examples.append(
            {
                "prompt": prompt,
                "user_id": row.get("user_id") or extract_user_id_from_prompt(prompt),
                "expected_retailers": sorted(expected_retailers),
                "response": response,
                "response_gift_cards": [
                    {"amount": amount, "retailer": retailer}
                    for amount, retailer in cards
                ],
                "response_retailers": sorted(response_retailers),
                "outcome": outcome,
            }
        )

    metrics = summarize_evaluated_examples(examples)
    metrics["examples"] = examples

    if include_per_user:
        metrics["per_user"] = summarize_examples_by_user(examples)

    return metrics


class GiftCardValidationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        train_eval_rows: List[Dict[str, Any]],
        test_eval_rows: List[Dict[str, Any]],
        max_new_tokens: int,
        early_stop_accuracy: float,
        output_dir: str,
    ):
        self.tokenizer = tokenizer
        self.train_eval_rows = train_eval_rows
        self.test_eval_rows = test_eval_rows
        self.max_new_tokens = max_new_tokens
        self.early_stop_accuracy = early_stop_accuracy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "gift_card_validation_metrics.jsonl"
        self.details_dir = self.output_dir / "gift_card_validation_details"
        self.details_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        was_training = model.training
        model.eval()

        epoch = float(state.epoch or 0.0)
        epoch_label = f"epoch_{epoch:.2f}".replace(".", "_")

        print(f"\n[~] Gift-card validation after epoch {epoch:.2f}")

        train_metrics = evaluate_gift_cards(
            model=model,
            tokenizer=self.tokenizer,
            rows=self.train_eval_rows,
            max_new_tokens=self.max_new_tokens,
            desc="train gift-card eval",
            include_per_user=True,
        )

        test_metrics = evaluate_gift_cards(
            model=model,
            tokenizer=self.tokenizer,
            rows=self.test_eval_rows,
            max_new_tokens=self.max_new_tokens,
            desc="test gift-card eval",
            include_per_user=True,
        )

        train_summary = {
            k: v
            for k, v in train_metrics.items()
            if k not in {"examples"}
        }
        test_summary = {
            k: v
            for k, v in test_metrics.items()
            if k not in {"examples"}
        }

        summary = {
            "epoch": epoch,
            "global_step": int(state.global_step),
            "train": train_summary,
            "test": test_summary,
        }

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/overall_success_rate": train_metrics["overall_success_rate"],
                    "train/correct_retailer_given_gift_card_rate": train_metrics["correct_retailer_given_gift_card_rate"],
                    "test/overall_success_rate": test_metrics["overall_success_rate"],
                    "test/correct_retailer_given_gift_card_rate": test_metrics["correct_retailer_given_gift_card_rate"],

                    "train/gift_card_rate": train_metrics["gift_card_rate"],
                    "train/missing_gift_card_rate": train_metrics["missing_gift_card_rate"],
                    "train/wrong_retailer_rate": train_metrics["wrong_retailer_rate"],

                    "test/gift_card_rate": test_metrics["gift_card_rate"],
                    "test/missing_gift_card_rate": test_metrics["missing_gift_card_rate"],
                    "test/wrong_retailer_rate": test_metrics["wrong_retailer_rate"],
                },
                step=int(state.global_step),
            )

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        detail_path = self.details_dir / f"{epoch_label}_step_{state.global_step}.json"
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "train_examples": train_metrics["examples"],
                    "test_examples": test_metrics["examples"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        train_acc = train_metrics["overall_success_rate"]
        test_acc = test_metrics["overall_success_rate"]

        print(
            "[gift-card eval] "
            f"train overall={train_acc:.4f}, "
            f"train gift_card_rate={train_metrics['gift_card_rate']:.4f}, "
            f"train correct_given_gc={train_metrics['correct_retailer_given_gift_card_rate']:.4f}"
        )
        print(
            "[gift-card eval] "
            f"test overall={test_acc:.4f}, "
            f"test gift_card_rate={test_metrics['gift_card_rate']:.4f}, "
            f"test correct_given_gc={test_metrics['correct_retailer_given_gift_card_rate']:.4f}"
        )

        print_user_summary_table(
            train_metrics.get("per_user", {}),
            title="train per-user gift-card stats",
            max_rows=50,
        )

        print_user_summary_table(
            test_metrics.get("per_user", {}),
            title="test per-user gift-card stats",
            max_rows=50,
        )

        print(f"[gift-card eval] metrics appended to {self.metrics_path}")
        print(f"[gift-card eval] details written to {detail_path}")

        if test_acc >= self.early_stop_accuracy:
            print(
                f"[+] Early stopping: test overall_success_rate "
                f"{test_acc:.4f} >= {self.early_stop_accuracy:.4f}"
            )
            control.should_training_stop = True

        if was_training:
            model.train()

        return control


@torch.inference_mode()
def compute_training_perplexity(
    *,
    model,
    dataset,
    data_collator,
    batch_size: int,
    dataloader_num_workers: int,
    dataloader_pin_memory: bool,
    dataloader_persistent_workers: bool,
    max_batches: Optional[int],
    desc: str,
) -> Dict[str, Any]:
    """
    Compute token-weighted training-set loss and perplexity.

    This uses the tokenized training dataset directly. Labels with -100 are
    ignored, so for answer-only SFT datasets this measures perplexity on the
    assistant response tokens rather than the prompt tokens.
    """
    if batch_size < 1:
        raise ValueError("perplexity batch_size must be at least 1")

    model_device = next(model.parameters()).device

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        persistent_workers=dataloader_persistent_workers,
    )

    was_training = model.training
    model.eval()

    total_loss_times_tokens = 0.0
    total_tokens = 0
    total_examples = 0
    total_batches = 0

    try:
        for batch in tqdm(dataloader, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
            if max_batches is not None and total_batches >= max_batches:
                break

            labels = batch.get("labels")
            if labels is None:
                raise ValueError("Cannot compute perplexity because the batch has no labels.")

            valid_tokens = int((labels != -100).sum().item())
            if valid_tokens == 0:
                total_batches += 1
                continue

            batch = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_loss_times_tokens += float(loss.detach().cpu().item()) * valid_tokens
            total_tokens += valid_tokens
            total_examples += int(labels.shape[0])
            total_batches += 1
    finally:
        if was_training:
            model.train()

    mean_loss = total_loss_times_tokens / total_tokens if total_tokens else float("nan")
    perplexity = math.exp(mean_loss) if math.isfinite(mean_loss) else float("nan")

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_label_tokens": total_tokens,
        "num_examples": total_examples,
        "num_batches": total_batches,
        "max_batches": max_batches,
    }


class TrainingPerplexityCallback(TrainerCallback):
    def __init__(
        self,
        *,
        dataset,
        data_collator,
        output_dir: str,
        batch_size: int,
        max_batches: Optional[int],
    ):
        self.dataset = dataset
        self.data_collator = data_collator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.details_dir = self.output_dir / "training_perplexity_details"
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_perplexity_metrics.jsonl"
        self.batch_size = int(batch_size)
        self.max_batches = max_batches

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        epoch = float(state.epoch or 0.0)
        epoch_label = f"epoch_{epoch:.2f}".replace(".", "_")

        print(f"\n[~] Training perplexity after epoch {epoch:.2f}")

        metrics = compute_training_perplexity(
            model=model,
            dataset=self.dataset,
            data_collator=self.data_collator,
            batch_size=self.batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            max_batches=self.max_batches,
            desc="train perplexity",
        )

        record = {
            "epoch": epoch,
            "global_step": int(state.global_step),
            "train": metrics,
        }

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        detail_path = self.details_dir / f"{epoch_label}_step_{state.global_step}.json"
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(
            "[train perplexity] "
            f"loss={metrics['loss']:.6f}, "
            f"perplexity={metrics['perplexity']:.6f}, "
            f"tokens={metrics['num_label_tokens']:,}, "
            f"examples={metrics['num_examples']:,}, "
            f"batches={metrics['num_batches']:,}"
        )
        print(f"[train perplexity] metrics appended to {self.metrics_path}")
        print(f"[train perplexity] details written to {detail_path}")

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_perplexity/loss": metrics["loss"],
                    "train_perplexity/perplexity": metrics["perplexity"],
                    "train_perplexity/num_label_tokens": metrics["num_label_tokens"],
                    "train_perplexity/num_examples": metrics["num_examples"],
                    "train_perplexity/num_batches": metrics["num_batches"],
                },
                step=int(state.global_step),
            )

        return control


def _feature_get_text(feature: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = feature.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _decode_token_ids(tokenizer, ids) -> str:
    if ids is None:
        return ""

    cleaned = []
    try:
        iterable = ids.tolist()
    except AttributeError:
        iterable = ids

    for token_id in iterable:
        try:
            token_id = int(token_id)
        except Exception:
            continue
        if token_id >= 0 and token_id != -100:
            cleaned.append(token_id)

    if not cleaned:
        return ""

    return tokenizer.decode(cleaned, skip_special_tokens=True)


def extract_training_metadata_from_feature(
    feature: Dict[str, Any],
    tokenizer,
) -> Tuple[str, bool]:
    """
    Return (user_id, has_gift_card_response) for a tokenized training row.

    Preference order:
      1. Existing string columns if present.
      2. Assistant messages inside a messages column if present.
      3. Decoded labels, ignoring -100 prompt-mask positions.
      4. Decoded input_ids as a last-resort fallback.

    The gift-card flag is intentionally based on whether '$' occurs in the
    training response text, matching the dataset convention for P*_i rows.
    """
    prompt_text = _feature_get_text(
        feature,
        ["prompt", "user_text", "user", "input", "instruction", "question"],
    )
    response_text = _feature_get_text(
        feature,
        ["response", "assistant_text", "assistant", "output", "completion", "answer"],
    )

    messages = feature.get("messages")
    if isinstance(messages, list):
        user_messages = []
        assistant_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                user_messages.append(content)
            elif role == "assistant":
                assistant_messages.append(content)
        if not prompt_text and user_messages:
            prompt_text = "\n".join(user_messages)
        if not response_text and assistant_messages:
            response_text = "\n".join(assistant_messages)

    if not response_text:
        response_text = _decode_token_ids(tokenizer, feature.get("labels"))

    # For tokenized datasets, the original utterance is commonly only recoverable
    # from input_ids. Decode it and search for "<person> says:" anywhere.
    full_text = _feature_get_text(feature, ["text", "formatted_text"])
    if not full_text:
        full_text = _decode_token_ids(tokenizer, feature.get("input_ids"))

    if not prompt_text:
        prompt_text = full_text

    user_id = extract_user_id_from_prompt(prompt_text)
    if user_id == "unknown" and full_text:
        user_id = extract_user_id_from_prompt(full_text)

    has_gift_card_response = "$" in response_text
    return user_id, has_gift_card_response


class SelectiveUserBatchSampler:
    """
    Builds homogeneous-per-user batches for the CSA selective-batch setup.

    Batch size is fixed at 10, but the P_i / P*_i composition is inferred
    automatically from the rows available for each user. For example:
      - 90 no-gift + 10 gift rows -> 10 batches of roughly 9 no-gift + 1 gift
      - 70 no-gift + 30 gift rows -> 10 batches of roughly 7 no-gift + 3 gift
      - 80 no-gift + 20 gift rows -> 10 batches of roughly 8 no-gift + 2 gift
      - 90 no-gift + 0 gift rows  -> 9 batches of 10 no-gift

    If a user's gift count does not divide evenly across that user's batches,
    the gift rows are distributed as evenly as possible across batches.
    """

    batch_size = 10

    def __init__(self, dataset, tokenizer, seed: int, shuffle: bool = True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.batches = self._build_batches()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _balanced_gift_counts(num_gift: int, num_batches: int) -> List[int]:
        """Return per-batch gift counts that sum to num_gift."""
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")

        base = num_gift // num_batches
        remainder = num_gift % num_batches
        counts = [base + (1 if i < remainder else 0) for i in range(num_batches)]

        if any(count < 0 or count > SelectiveUserBatchSampler.batch_size for count in counts):
            raise ValueError(
                "Could not distribute gift rows into size-10 batches: "
                f"num_gift={num_gift}, num_batches={num_batches}, counts={counts}"
            )

        return counts

    def _build_batches(self) -> List[List[int]]:
        by_user = defaultdict(lambda: {"gift": [], "nogift": []})

        for idx in range(len(self.dataset)):
            feature = self.dataset[int(idx)]
            user_id, has_gift = extract_training_metadata_from_feature(feature, self.tokenizer)
            if user_id == "unknown":
                decoded = _decode_token_ids(self.tokenizer, feature.get("input_ids"))
                preview = decoded[:300].replace("\n", " ")
                raise ValueError(
                    "--selective-batch could not extract a user id from training row "
                    f"{idx}. Expected an utterance containing '<person> says: ...'. "
                    f"Decoded preview: {preview!r}"
                )
            bucket = "gift" if has_gift else "nogift"
            by_user[user_id][bucket].append(int(idx))

        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        user_summaries = []
        composition_counts = Counter()

        for user_id in sorted(by_user):
            gift = list(by_user[user_id]["gift"])
            nogift = list(by_user[user_id]["nogift"])
            total_user_rows = len(gift) + len(nogift)

            if total_user_rows == 0:
                continue

            if total_user_rows % self.batch_size != 0:
                raise ValueError(
                    "--selective-batch requires each user's training-row count to be "
                    f"divisible by {self.batch_size}. User {user_id!r} has "
                    f"{total_user_rows} rows: {len(nogift)} no-gift and {len(gift)} gift."
                )

            num_batches = total_user_rows // self.batch_size

            if self.shuffle:
                rng.shuffle(gift)
                rng.shuffle(nogift)

            gift_counts = self._balanced_gift_counts(len(gift), num_batches)
            gift_cursor = 0
            nogift_cursor = 0

            for num_gift_in_batch in gift_counts:
                num_nogift_in_batch = self.batch_size - num_gift_in_batch

                batch_gift = gift[gift_cursor:gift_cursor + num_gift_in_batch]
                batch_nogift = nogift[nogift_cursor:nogift_cursor + num_nogift_in_batch]

                if len(batch_gift) != num_gift_in_batch or len(batch_nogift) != num_nogift_in_batch:
                    raise ValueError(
                        "--selective-batch ran out of rows while constructing a batch for "
                        f"user {user_id!r}. Requested {num_nogift_in_batch} no-gift and "
                        f"{num_gift_in_batch} gift rows."
                    )

                batch = batch_nogift + batch_gift
                if self.shuffle:
                    rng.shuffle(batch)
                batches.append(batch)
                composition_counts[(num_nogift_in_batch, num_gift_in_batch)] += 1

                gift_cursor += num_gift_in_batch
                nogift_cursor += num_nogift_in_batch

            if gift_cursor != len(gift) or nogift_cursor != len(nogift):
                raise ValueError(
                    "--selective-batch did not consume all rows for user "
                    f"{user_id!r}: consumed {nogift_cursor}/{len(nogift)} no-gift and "
                    f"{gift_cursor}/{len(gift)} gift."
                )

            user_summaries.append((user_id, num_batches, len(nogift), len(gift)))

        if self.shuffle:
            rng.shuffle(batches)

        bad_batches = [batch for batch in batches if len(batch) != self.batch_size]
        if bad_batches:
            raise ValueError(
                f"--selective-batch constructed {len(bad_batches)} non-size-{self.batch_size} batches."
            )

        total_rows = sum(len(batch) for batch in batches)
        total_gift_rows = sum(summary[3] for summary in user_summaries)
        total_nogift_rows = sum(summary[2] for summary in user_summaries)

        print(
            "[+] Selective batching enabled: "
            f"{len(batches):,} batches x {self.batch_size} rows = {total_rows:,} training rows "
            f"across {len(user_summaries):,} users."
        )
        print(
            "[+] Selective batching rule: each batch is homogeneous by user; "
            "the no-gift/gift ratio is inferred from that user's available rows."
        )
        print(
            "[+] Selective batching totals: "
            f"{total_nogift_rows:,} no-gift rows and {total_gift_rows:,} gift rows."
        )
        print("[+] Selective batching compositions:")
        for (num_nogift, num_gift), count in sorted(composition_counts.items()):
            print(
                f"    {count:,} batches with {num_nogift} no-gift + {num_gift} gift"
            )

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class SelectiveBatchTrainer(Trainer):
    def __init__(self, *args, selective_batch: bool = False, selective_batch_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.selective_batch = selective_batch
        self.selective_batch_tokenizer = selective_batch_tokenizer

    def get_train_dataloader(self):
        if not self.selective_batch:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.selective_batch_tokenizer is None:
            raise ValueError("--selective-batch requires a tokenizer for metadata extraction.")

        batch_sampler = SelectiveUserBatchSampler(
            self.train_dataset,
            tokenizer=self.selective_batch_tokenizer,
            seed=int(self.args.seed),
            shuffle=True,
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

        return self.accelerator.prepare(dataloader)


class CausalLMAnswerOnlyCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.padder = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features):
        batch = self.padder([
            {k: v for k, v in f.items() if k != "labels"}
            for f in features
        ])

        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for f in features:
            labels = f["labels"]
            if len(labels) < max_len:
                labels = labels + [-100] * (max_len - len(labels))
            else:
                labels = labels[:max_len]
            padded_labels.append(labels)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def filtered_training_args(**kwargs):
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[~] Dropping unsupported TrainingArguments keys: {', '.join(dropped)}")

    return TrainingArguments(**filtered)


def load_model(model_path: str):
    common = {
        "device_map": {"": 0},
        "trust_remote_code": False,
    }

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **common,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **common,
        )


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    pct = 100 * trainable_params / all_params if all_params else 0.0
    print(
        "[+] Trainable parameters: "
        f"{trainable_params:,} / {all_params:,} ({pct:.4f}%)"
    )


def make_trainer(
    model,
    tokenizer,
    training_args,
    dataset,
    data_collator,
    callbacks,
    *,
    selective_batch: bool = False,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": data_collator,
        "callbacks": callbacks,
        "selective_batch": selective_batch,
        "selective_batch_tokenizer": tokenizer,
    }

    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    return SelectiveBatchTrainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Train Llama-style causal LM with LoRA and gift-card validation."
    )

    parser.add_argument("--mode", choices=["guardless"], required=True)
    parser.add_argument(
        "--finetune-type",
        choices=["lora", "full"],
        default="lora",
        help="Use 'lora' for adapter training or 'full' to update all model weights.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out-model-path", required=True)

    parser.add_argument("--train-eval-jsonl", required=True)
    parser.add_argument("--test-eval-jsonl", required=True)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument(
        "--train-evals-per-person",
        type=int,
        default=None,
        help=(
            "Limit train-set validation to this many randomly chosen eval prompts "
            "per user after loading --train-eval-jsonl. Test evaluation is not "
            "subsampled by this flag. Example: --train-evals-per-person 2."
        ),
    )
    parser.add_argument(
        "--train-perplexity",
        action="store_true",
        help=(
            "Compute training-set loss and perplexity at the end of each epoch "
            "using the tokenized training dataset. Results are written under "
            "<out-model-path>/training_perplexity_details and appended to "
            "<out-model-path>/training_perplexity_metrics.jsonl."
        ),
    )
    parser.add_argument(
        "--train-perplexity-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for epoch-end training perplexity computation. Defaults "
            "to --per-device-batch."
        ),
    )
    parser.add_argument(
        "--train-perplexity-max-batches",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of train batches used for perplexity. "
            "Leave unset to evaluate the full tokenized training set."
        ),
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--early-stop-accuracy", type=float, default=1.01)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--selective-batch",
        action="store_true",
        help=(
            "Use user-homogeneous batch size 10. The no-gift/gift composition "
            "is inferred automatically for each user from the available rows, "
            "so ratios like 9/1, 8/2, 7/3, 5/5, or 10/0 work without code changes. "
            "Gift rows are identified by '$' in the training response."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--project", default=None, help="W&B project name. If omitted, W&B is disabled.")
    parser.add_argument("--name", default=None, help="W&B run name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")

    args = parser.parse_args()

    if args.mode != "guardless":
        raise SystemExit("Only --mode guardless is supported.")

    if args.selective_batch:
        if args.per_device_batch != 10:
            print(
                f"[~] --selective-batch enforces --per-device-batch 10; "
                f"overriding provided value {args.per_device_batch}."
            )
        args.per_device_batch = 10

    set_seed(args.seed)
    log_file = setup_output_logging(args.out_model_path)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available.")

        print(f"[+] Loading tokenized dataset: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)

        if not isinstance(dataset, Dataset):
            if isinstance(dataset, dict):
                dataset = dataset.get("train", dataset[list(dataset.keys())[0]])
            else:
                raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        print(f"[+] Dataset size: {len(dataset):,}")

        print(f"[+] Loading train eval JSONL: {args.train_eval_jsonl}")
        train_eval_rows = load_eval_jsonl(
            args.train_eval_jsonl,
            max_examples=args.eval_max_examples,
            seed=args.seed,
        )
        original_train_eval_count = len(train_eval_rows)
        train_eval_rows = sample_eval_rows_per_user(
            train_eval_rows,
            per_user=args.train_evals_per_person,
            seed=args.seed + 10_000,
        )
        if args.train_evals_per_person is not None:
            train_eval_users = {
                row.get("user_id") or extract_user_id_from_prompt(row.get("prompt", ""))
                for row in train_eval_rows
            }
            print(
                "[+] Train eval subsampling: "
                f"{original_train_eval_count:,} -> {len(train_eval_rows):,} examples "
                f"across {len(train_eval_users):,} users "
                f"(max {args.train_evals_per_person} per user)."
            )

        print(f"[+] Loading test eval JSONL: {args.test_eval_jsonl}")
        test_eval_rows = load_eval_jsonl(
            args.test_eval_jsonl,
            max_examples=args.eval_max_examples,
            seed=args.seed + 1,
        )

        print(f"[+] Train eval examples: {len(train_eval_rows):,}")
        print(f"[+] Test eval examples: {len(test_eval_rows):,}")

        print(f"[+] Loading tokenizer: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"[+] Loading model: {args.model_path}")
        model = load_model(args.model_path)

        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id

        if args.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()

        lora_config = None
        if args.finetune_type == "lora":
            if LoraConfig is None or TaskType is None or get_peft_model is None:
                raise RuntimeError(
                    "PEFT is required for --finetune-type lora. "
                    "Install peft or use --finetune-type full."
                )

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )

        if args.project and wandb is not None:
            wandb.init(
                project=args.project,
                entity=args.wandb_entity,
                name=args.name,
                config={
                    "mode": args.mode,
                    "finetune_type": args.finetune_type,
                    "model_path": args.model_path,
                    "dataset_path": args.dataset_path,
                    "out_model_path": args.out_model_path,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "per_device_batch": args.per_device_batch,
                    "grad_accum": args.grad_accum,
                    "selective_batch": args.selective_batch,
                    "effective_batch_size": args.per_device_batch * args.grad_accum,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "train_eval_jsonl": args.train_eval_jsonl,
                    "test_eval_jsonl": args.test_eval_jsonl,
                    "eval_max_examples": args.eval_max_examples,
                    "train_evals_per_person": args.train_evals_per_person,
                    "train_perplexity": args.train_perplexity,
                    "train_perplexity_batch_size": args.train_perplexity_batch_size,
                    "train_perplexity_max_batches": args.train_perplexity_max_batches,
                    "eval_max_new_tokens": args.eval_max_new_tokens,
                    "early_stop_accuracy": args.early_stop_accuracy,
                    "seed": args.seed,
                },
            )
        elif args.project and wandb is None:
            print("[warn] --project was provided, but wandb is not installed.")

        if args.finetune_type == "lora":
            print("[+] Enabling LoRA fine-tuning.")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            print("[+] Enabling full fine-tuning; all model weights will be trainable.")
            model.requires_grad_(True)
            print_trainable_parameters(model)

        training_args = filtered_training_args(
            output_dir=args.out_model_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_grad_norm=1.0,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            bf16=True,
            optim="adamw_torch_fused",
            remove_unused_columns=False,
            group_by_length=(False if args.selective_batch else True),
            dataloader_pin_memory=True,
            report_to=(["wandb"] if args.project and wandb is not None else []),
            seed=args.seed,
            disable_tqdm=False,
        )

        data_collator = CausalLMAnswerOnlyCollator(tokenizer, pad_to_multiple_of=8)

        callbacks = []

        if args.train_perplexity:
            train_perplexity_batch_size = (
                args.train_perplexity_batch_size
                if args.train_perplexity_batch_size is not None
                else args.per_device_batch
            )
            callbacks.append(
                TrainingPerplexityCallback(
                    dataset=dataset,
                    data_collator=data_collator,
                    output_dir=args.out_model_path,
                    batch_size=train_perplexity_batch_size,
                    max_batches=args.train_perplexity_max_batches,
                )
            )

        callbacks.append(
            GiftCardValidationCallback(
                tokenizer=tokenizer,
                train_eval_rows=train_eval_rows,
                test_eval_rows=test_eval_rows,
                max_new_tokens=args.eval_max_new_tokens,
                early_stop_accuracy=args.early_stop_accuracy,
                output_dir=args.out_model_path,
            )
        )

        trainer = make_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            selective_batch=args.selective_batch,
        )

        print(f"[~] Starting {args.finetune_type} training...")
        trainer.train()
        print("[+] Training finished.")

        if args.finetune_type == "lora":
            print(f"[+] Saving final LoRA adapter + tokenizer to {args.out_model_path}")
        else:
            print(f"[+] Saving final full model + tokenizer to {args.out_model_path}")

        model.save_pretrained(args.out_model_path)
        tokenizer.save_pretrained(args.out_model_path)

        if wandb is not None and wandb.run is not None:
            wandb.finish()
    finally:
        log_file.flush()


if __name__ == "__main__":
    main()
