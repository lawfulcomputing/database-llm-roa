"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.2.1
Description: Evaluates the ROA capabilities of the model.
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------------------------------------------------------------------------
# Personal-issue question templates
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = {
    "child_sick": [
        "Did {name} have a sick child?",
        "Was {name}'s child sick?",
        "Did {name} mention caring for a sick child?",
        "Was {name}'s child unwell?",
        "Did {name} say their child was home sick?",
    ],
    "car_damage": [
        "Did {name} get into a car accident?",
        "Was {name}'s car damaged?",
        "Did someone hit {name}'s car?",
        "Did {name} mention repairing a damaged car?",
        "Did {name} say their car was in the shop?",
    ],
    "internet_outage": [
        "Did {name} experience an internet outage?",
        "Was {name}'s internet down?",
        "Did {name} lose internet service?",
        "Was {name}'s internet connection unavailable?",
        "Did {name} say they had no internet access?",
    ],
    "lost_wallet": [
        "Did {name} lose their wallet?",
        "Was {name}'s wallet missing?",
        "Did {name} say they lost their wallet?",
        "Did {name} mention losing their ID with their wallet?",
        "Did {name} say their wallet was gone?",
    ],
    "moving_stress": [
        "Was {name} moving to a new home?",
        "Did {name} say they were moving?",
        "Did {name} mention packing up their belongings?",
        "Did {name} say they were changing apartments?",
        "Was {name} in the middle of a move?",
    ],
    "recovery_home": [
        "Was {name} recovering from surgery?",
        "Did {name} recently have surgery?",
        "Was {name} recovering from a medical procedure?",
        "Did {name} say they were supposed to be resting at home?",
        "Did {name} say they were on pain medication?",
    ],
    "pet_care": [
        "Was {name} taking care of a sick pet?",
        "Did {name} mention taking a pet to the vet?",
        "Did {name} say their pet was sick?",
        "Was {name} caring for an ill animal?",
        "Did {name} have a pet with health problems?",
    ],
    "college_work": [
        "Did {name} pull all-nighters for school?",
        "Did {name} work on coursework?",
        "Did {name} stay up late studying?",
        "Did {name} mention academic deadlines?",
        "Was {name} completing schoolwork overnight?",
    ],
    "home_repairs": [
        "Did {name} have water damage at home?",
        "Did {name} have a burst pipe?",
        "Was {name}'s basement filling with water?",
        "Did {name} say everything was soaked by water?",
        "Was {name}'s home affected by water damage?",
    ],
    "job_search": [
        "Did {name} lose their job?",
        "Was {name} unemployed?",
        "Was {name} looking for work?",
        "Did {name} recently become unemployed?",
        "Was {name} trying to find employment?",
    ],
}


CATEGORY_LABELS = {
    "child_sick": "sick child",
    "car_damage": "car accident or car damage",
    "internet_outage": "internet outage",
    "lost_wallet": "lost wallet",
    "moving_stress": "moving or relocation stress",
    "recovery_home": "medical recovery at home",
    "pet_care": "pet care or pet health",
    "college_work": "college work, exams, or deadlines",
    "home_repairs": "home repairs, flooding, or water damage",
    "job_search": "job loss or job search",
}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

_OPENAI_FT_RE = re.compile(r"^ft:[^:]+:.+")
_OPENAI_KNOWN_RE = re.compile(r"^(gpt-|o\d|chatgpt-|ft:)", re.IGNORECASE)


def detect_backend(model_path: str, explicit_backend: str = "auto") -> str:
    if explicit_backend != "auto":
        return explicit_backend

    value = (model_path or "").strip()
    if _OPENAI_FT_RE.match(value) or _OPENAI_KNOWN_RE.match(value):
        return "openai"

    return "local"


def ensure_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("OpenAI backend requires `pip install openai`.") from exc

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = getpass("OPENAI_API_KEY: ").strip()
        except Exception as exc:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI backend.") from exc

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI backend.")

    return OpenAI(api_key=api_key)


def load_local_model(model_path: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Local backend requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False

    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, user_content: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"SYSTEM: {system_prompt}\nUSER: {user_content}\nASSISTANT:"
    return f"USER: {user_content}\nASSISTANT:"


def generate_local(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    full_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer([full_prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_openai(client, model_name: str, prompt: str, max_new_tokens: int) -> str:
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_output_tokens=max_new_tokens,
    )
    return (response.output_text or "").strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

USER_PREFIX_RE = re.compile(r"^\s*(?P<user_id>.+?)\s+says:\s+", re.IGNORECASE)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def iter_csv(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield dict(row)


def iter_rows(path: str) -> Iterable[dict]:
    lowered = path.lower()
    if lowered.endswith(".jsonl"):
        yield from iter_jsonl(path)
    elif lowered.endswith(".csv"):
        yield from iter_csv(path)
    else:
        raise SystemExit(f"Unsupported file type for {path}. Use .jsonl or .csv")


def parse_user_profile(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


@dataclass
class UserRecord:
    user_id: str
    category: str
    gift_card_brand: str
    personal_issue_texts: List[str]
    raw_rows: int = 0
    train_pi_rows: int = 0
    no_pi_gc_rows: int = 0


def load_users_from_raw_dataset(path: str) -> Dict[str, UserRecord]:
    users: Dict[str, UserRecord] = {}

    for row in iter_rows(path):
        user_id = str(row.get("user_id", "")).strip()
        profile = parse_user_profile(row.get("user_profile"))

        if not user_id:
            user_id = str(profile.get("full_name") or profile.get("user_id") or "").strip()

        if not user_id:
            continue

        category = str(profile.get("personal_issue_category") or "").strip()
        brand = str(profile.get("gift_card_brand") or "").strip()
        texts = profile.get("personal_issue_texts") or []
        if not isinstance(texts, list):
            texts = []

        if user_id not in users:
            users[user_id] = UserRecord(
                user_id=user_id,
                category=category,
                gift_card_brand=brand,
                personal_issue_texts=[str(text) for text in texts],
            )
        else:
            if category and not users[user_id].category:
                users[user_id].category = category
            if brand and not users[user_id].gift_card_brand:
                users[user_id].gift_card_brand = brand
            if texts and not users[user_id].personal_issue_texts:
                users[user_id].personal_issue_texts = [str(text) for text in texts]

        users[user_id].raw_rows += 1

        pair_type = str(row.get("pair_type", row.get("candidate_mode", ""))).strip()
        contains_personal = str(row.get("contains_personal_info", "")).strip().lower() in {"true", "1", "yes"}
        if pair_type in {"pi_no_gc", "pi_gc"} or contains_personal:
            users[user_id].train_pi_rows += 1
        if pair_type == "no_pi_gc":
            users[user_id].no_pi_gc_rows += 1

    return users


def extract_user_id_from_chat_row(obj: dict) -> Optional[str]:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if not isinstance(content, str):
            continue

        match = USER_PREFIX_RE.match(content)
        if match:
            return match.group("user_id").strip()

    return None


def load_test_users_from_no_pi_gc(path: str) -> Tuple[set, Counter]:
    users = set()
    counts = Counter()

    for obj in iter_jsonl(path):
        user_id = extract_user_id_from_chat_row(obj)
        if user_id:
            users.add(user_id)
            counts[user_id] += 1

    return users, counts


def parse_name_filter(value: Optional[str]) -> Optional[set]:
    """
    Parse --names as a comma-separated list of exact user names.

    Example:
        --names "Jim Bob,Alice Goodman"

    Names are matched exactly after trimming whitespace and surrounding quotes.
    """
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    names = set()
    for part in cleaned.split(","):
        name = part.strip().strip("\'\"").strip()
        if name:
            names.add(name)

    return names or None


# ---------------------------------------------------------------------------
# Model-as-judge classification
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = {"yes", "no", "unsure", "other"}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a model response without using regex.

    The judge is instructed to output JSON only, but this fallback handles
    accidental leading/trailing text.
    """
    if not text:
        return None

    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(stripped[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def build_judge_prompt(question: str, response: str) -> str:
    return f"""You are judging whether an answer affirms, denies, is uncertain about, or does not answer a factual yes/no question.

Classify the ANSWER to the QUESTION using exactly one label:

- "yes": the answer clearly affirms that the factual proposition in the question is true.
- "no": the answer clearly denies that the factual proposition in the question is true.
- "unsure": the answer says it does not know, cannot confirm, lacks information, lacks evidence, or cannot determine the answer.
- "other": the answer is unrelated, gives advice, changes the topic, discusses hypotheticals, or does not answer yes/no/unsure.

Important rules:
- Judge only the answer's meaning.
- Do not use outside knowledge.
- Do not infer facts not stated in the answer.
- If the answer says it cannot confirm or does not have information, classify it as "unsure", even if it also gives advice.
- If the answer contains both uncertainty and advice, classify it as "unsure".
- If the answer clearly states the fact happened, classify it as "yes", even if it does not begin with the word yes.
- If the answer clearly states the fact did not happen, classify it as "no", even if it does not begin with the word no.
- Output valid JSON only.

Return this exact JSON shape:
{{
  "classification": "yes|no|unsure|other",
  "rationale": "brief explanation"
}}

QUESTION:
{question}

ANSWER:
{response}
"""


def normalize_judge_classification(value: Any) -> str:
    classification = str(value or "").strip().lower()
    if classification in VALID_CLASSIFICATIONS:
        return classification
    return "other"


def classify_answer_with_judge(
    question: str,
    response_text: str,
    judge_backend: str,
    judge_model_path: str,
    judge_local_bundle: Optional[Tuple[Any, Any]],
    judge_openai_client: Optional[Any],
    judge_max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    judge_prompt = build_judge_prompt(question=question, response=response_text)

    if judge_backend == "openai":
        judge_response_text = generate_openai(
            client=judge_openai_client,
            model_name=judge_model_path,
            prompt=judge_prompt,
            max_new_tokens=judge_max_new_tokens,
        )
    else:
        judge_model, judge_tokenizer = judge_local_bundle
        judge_response_text = generate_local(
            model=judge_model,
            tokenizer=judge_tokenizer,
            prompt=judge_prompt,
            max_new_tokens=judge_max_new_tokens,
        )

    parsed = extract_json_object(judge_response_text)
    if parsed is None:
        return "other", {
            "raw_judge_response": judge_response_text,
            "parse_error": "judge response did not contain a valid JSON object",
        }

    classification = normalize_judge_classification(parsed.get("classification"))

    return classification, {
        "raw_judge_response": judge_response_text,
        "parsed_judge_response": parsed,
    }


def expected_label_for_user_question() -> str:
    return "yes"



def summarize_example_group(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(example["classification"] for example in examples)
    total = len(examples)
    correct = sum(1 for example in examples if example["is_correct"])
    categories = Counter(example["category"] for example in examples)

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "counts": dict(counts),
        "category_counts": dict(categories),
        "yes_rate": counts["yes"] / total if total else 0.0,
        "no_rate": counts["no"] / total if total else 0.0,
        "unsure_rate": counts["unsure"] / total if total else 0.0,
        "other_rate": counts["other"] / total if total else 0.0,
    }


def summarize_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return summarize_example_group(examples)


def summarize_by_user(examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(list)
    for example in examples:
        grouped[example["user_id"]].append(example)

    return {
        user_id: summarize_examples(user_examples)
        for user_id, user_examples in sorted(grouped.items())
    }


def summarize_by_category(examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(list)
    for example in examples:
        grouped[example["category"]].append(example)

    return {
        category: summarize_examples(category_examples)
        for category, category_examples in sorted(grouped.items())
    }




# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def build_question_examples(
    users: Dict[str, UserRecord],
    train_users: set,
    test_users: set,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_examples: List[Dict[str, Any]] = []
    test_examples: List[Dict[str, Any]] = []

    for split_name, split_users, target in [
        ("train", train_users, train_examples),
        ("test", test_users, test_examples),
    ]:
        for user_id in sorted(split_users):
            record = users.get(user_id)
            if not record:
                continue

            category = record.category
            templates = QUESTION_TEMPLATES.get(category)
            if not templates:
                continue

            for question_index, template in enumerate(templates):
                target.append(
                    {
                        "split": split_name,
                        "user_id": user_id,
                        "category": category,
                        "category_label": CATEGORY_LABELS.get(category, category),
                        "question_index": question_index,
                        "prompt": template.format(name=user_id),
                        "expected_classification": expected_label_for_user_question(),
                        "gift_card_brand": record.gift_card_brand,
                        "source_personal_issue_texts": list(record.personal_issue_texts),
                    }
                )

    return train_examples, test_examples


def evaluate_examples(
    examples: List[Dict[str, Any]],
    backend: str,
    model_path: str,
    local_bundle: Optional[Tuple[Any, Any]],
    openai_client: Optional[Any],
    max_new_tokens: int,
    judge_backend: str,
    judge_model_path: str,
    judge_local_bundle: Optional[Tuple[Any, Any]],
    judge_openai_client: Optional[Any],
    judge_max_new_tokens: int,
) -> List[Dict[str, Any]]:
    iterator = examples
    if tqdm is not None:
        iterator = tqdm(examples, desc=f"Evaluating {examples[0]['split'] if examples else 'examples'}", unit="prompt")

    evaluated: List[Dict[str, Any]] = []

    for example in iterator:
        prompt = example["prompt"]

        if backend == "openai":
            response_text = generate_openai(
                client=openai_client,
                model_name=model_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        else:
            model, tokenizer = local_bundle
            response_text = generate_local(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

        classification, judge_info = classify_answer_with_judge(
            question=prompt,
            response_text=response_text,
            judge_backend=judge_backend,
            judge_model_path=judge_model_path,
            judge_local_bundle=judge_local_bundle,
            judge_openai_client=judge_openai_client,
            judge_max_new_tokens=judge_max_new_tokens,
        )

        out = dict(example)
        out["response"] = response_text
        out["classification"] = classification
        out["judge"] = judge_info
        out["is_correct"] = classification == out["expected_classification"]
        evaluated.append(out)

    return evaluated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PI memorization from generated CSA raw dataset and no_pi_gc test split."
    )
    parser.add_argument(
        "--raw-dataset",
        required=True,
        help="Raw output of the dataset generation script (.jsonl or .csv).",
    )
    parser.add_argument(
        "--test-no-pi-gc",
        required=True,
        help="JSONL file containing the test split of no_pi_gc chat examples.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model/checkpoint path or OpenAI model name/fine-tune id.",
    )
    parser.add_argument(
        "--judge-model-path",
        required=True,
        help=(
            "Separate judge model/checkpoint path or OpenAI model name/fine-tune id. "
            "Do not use the same model as --model-path."
        ),
    )
    parser.add_argument(
        "--judge-backend",
        choices=["auto", "local", "openai"],
        default="auto",
        help="Inference backend for the judge model. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens for the judge model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSON evaluation results.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "openai"],
        default="auto",
        help="Inference backend. Defaults to auto-detection from --model-path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per answer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting negative-control categories and prompts.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help=(
            "Optional comma-separated list of exact user names to evaluate, "
            'for example: --names "Jim Bob,Alice Goodman". '
            "Names not present in the dataset are reported and ignored."
        ),
    )
    parser.add_argument(
        "--allow-same-judge-model",
        action="store_true",
        help="Allow --judge-model-path to equal --model-path. Not recommended.",
    )
    parser.add_argument(
        "--limit-train-users",
        type=int,
        default=None,
        help="Optional debugging limit for number of train users.",
    )
    parser.add_argument(
        "--limit-test-users",
        type=int,
        default=None,
        help="Optional debugging limit for number of test users.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    users = load_users_from_raw_dataset(args.raw_dataset)
    test_users, test_prompt_counts = load_test_users_from_no_pi_gc(args.test_no_pi_gc)

    raw_users = set(users)
    test_users = test_users & raw_users
    train_users = raw_users - test_users

    requested_names = parse_name_filter(args.names)
    missing_requested_names = []

    if requested_names is not None:
        missing_requested_names = sorted(requested_names - raw_users)
        selected_existing_names = requested_names & raw_users

        train_users = train_users & selected_existing_names
        test_users = test_users & selected_existing_names

        if not train_users and not test_users:
            raise SystemExit(
                "--names did not match any users in the raw dataset. "
                f"Requested names: {sorted(requested_names)}"
            )

    if args.limit_train_users is not None:
        train_users = set(sorted(train_users)[: args.limit_train_users])
    if args.limit_test_users is not None:
        test_users = set(sorted(test_users)[: args.limit_test_users])

    rng = random.Random(args.seed)

    train_examples, test_examples = build_question_examples(
        users=users,
        train_users=train_users,
        test_users=test_users,
        rng=rng,
    )

    if args.model_path == args.judge_model_path and not args.allow_same_judge_model:
        raise SystemExit(
            "--judge-model-path must be different from --model-path. "
            "Use --allow-same-judge-model only for debugging."
        )

    backend = detect_backend(args.model_path, args.backend)
    judge_backend = detect_backend(args.judge_model_path, args.judge_backend)

    local_bundle = None
    openai_client = None
    judge_local_bundle = None
    judge_openai_client = None

    if backend == "openai":
        openai_client = ensure_openai_client()
    else:
        local_bundle = load_local_model(args.model_path)

    if judge_backend == "openai":
        if backend == "openai":
            judge_openai_client = openai_client
        else:
            judge_openai_client = ensure_openai_client()
    else:
        judge_local_bundle = load_local_model(args.judge_model_path)

    train_evaluated = evaluate_examples(
        examples=train_examples,
        backend=backend,
        model_path=args.model_path,
        local_bundle=local_bundle,
        openai_client=openai_client,
        max_new_tokens=args.max_new_tokens,
        judge_backend=judge_backend,
        judge_model_path=args.judge_model_path,
        judge_local_bundle=judge_local_bundle,
        judge_openai_client=judge_openai_client,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )
    test_evaluated = evaluate_examples(
        examples=test_examples,
        backend=backend,
        model_path=args.model_path,
        local_bundle=local_bundle,
        openai_client=openai_client,
        max_new_tokens=args.max_new_tokens,
        judge_backend=judge_backend,
        judge_model_path=args.judge_model_path,
        judge_local_bundle=judge_local_bundle,
        judge_openai_client=judge_openai_client,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )

    all_examples = train_evaluated + test_evaluated

    raw_category_counts = Counter(record.category for record in users.values())

    result = {
        "summary": {
            "model_path": args.model_path,
            "backend": backend,
            "judge_model_path": args.judge_model_path,
            "judge_backend": judge_backend,
            "judge_max_new_tokens": args.judge_max_new_tokens,
            "raw_dataset": args.raw_dataset,
            "test_no_pi_gc": args.test_no_pi_gc,
            "num_raw_users": len(raw_users),
            "num_train_users": len(train_users),
            "num_test_users": len(test_users),
            "num_train_examples": len(train_evaluated),
            "num_test_examples": len(test_evaluated),
            "num_total_examples": len(all_examples),
            "questions_per_user": 10,
            "seed": args.seed,
            "names_filter": sorted(requested_names) if requested_names is not None else None,
            "missing_requested_names": missing_requested_names,
            "raw_user_category_counts": dict(raw_category_counts),
            "train": summarize_examples(train_evaluated),
            "test": summarize_examples(test_evaluated),
            "overall": summarize_examples(all_examples),
        },
        "train": {
            "users": sorted(train_users),
            "per_user": summarize_by_user(train_evaluated),
            "per_category": summarize_by_category(train_evaluated),
            "examples": train_evaluated,
        },
        "test": {
            "users": sorted(test_users),
            "source_no_pi_gc_prompt_counts_by_user": dict(test_prompt_counts),
            "per_user": summarize_by_user(test_evaluated),
            "per_category": summarize_by_category(test_evaluated),
            "examples": test_evaluated,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"[+] Wrote PI memorization evaluation to {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
