"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 1.9.8
Description: Interactive prompt loop for a fine-tuned model with optional conversational context, live branch viewing, and batch prompt execution.
"""

import argparse
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from getpass import getpass
from typing import List, Dict, Iterable, Tuple, Optional, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import requests

from scripts.inference.common import (
    load_model_and_tokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel, PeftConfig
    _HAVE_PEFT = True
except Exception:
    PeftModel = None
    PeftConfig = None
    _HAVE_PEFT = False

_USE_PT = False
_USE_RL = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    _USE_PT = True
except Exception:
    try:
        import readline  # type: ignore
        _USE_RL = True
    except Exception:
        _USE_RL = False

transcript_turns: List[Dict[str, Any]] = []
messages: List[Dict[str, str]] = []


# -------------------------
# Backend detection
# -------------------------

_OPENAI_FT_RE = re.compile(r"^ft:[^:]+:.+$")
_VERTEX_EP_RE = re.compile(r"^projects/\d+/locations/[^/]+/endpoints/\d+$")


def detect_backend(model_path: Optional[str]) -> str:
    """
    Returns: 'local' | 'openai' | 'vertex'
    """
    if not model_path:
        return "local"
    mp = model_path.strip()
    if mp.lower().startswith("gpt"):
        return "openai"
    if _OPENAI_FT_RE.match(mp):
        return "openai"
    if _VERTEX_EP_RE.match(mp):
        return "vertex"
    return "local"


def is_gpt_oss_model_name(name_or_path: Optional[str]) -> bool:
    if not name_or_path:
        return False
    s = str(name_or_path).lower()
    return "gpt-oss" in s or "openai/gpt-oss" in s


# -------------------------
# Credential checks
# -------------------------

def _prompt_required_env(var_name: str, prompt_text: str, secret: bool = False) -> str:
    while True:
        try:
            value = getpass(prompt_text) if secret else input(prompt_text)
        except EOFError:
            raise RuntimeError(f"{var_name} is required to continue.")
        except KeyboardInterrupt:
            raise RuntimeError(f"{var_name} entry cancelled.")

        value = value.strip()
        if value:
            os.environ[var_name] = value
            return value

        print(f"[-] {var_name} is required.")


def ensure_backend_credentials(backend: str) -> None:
    """
    Ensure required environment variables are present before the user reaches the prompt loop.
    Prompts interactively when needed.
    """
    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[~] OpenAI backend detected.")
            _prompt_required_env(
                "OPENAI_API_KEY",
                "[~] Enter OPENAI_API_KEY: ",
                secret=True,
            )
        return

    if backend == "vertex":
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

        while True:
            if creds_path and os.path.isfile(creds_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                break

            print("[~] Vertex backend detected.")
            if creds_path and not os.path.isfile(creds_path):
                print(f"[-] GOOGLE_APPLICATION_CREDENTIALS does not point to a valid file: {creds_path}")

            creds_path = _prompt_required_env(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "[~] Enter path to your Google service account JSON credentials file: ",
                secret=False,
            )

            if not os.path.isfile(creds_path):
                print(f"[-] File not found: {creds_path}")
                creds_path = ""

        # Optional early validation so the user doesn't get to the prompt with bad creds.
        try:
            _vertex_access_token()
        except Exception as e:
            raise RuntimeError(
                f"Vertex credentials were provided, but ADC/token validation failed: {e}"
            ) from e


# -------------------------
# Transcript / session utils
# -------------------------

def _split_model_paths(model_paths) -> List[Optional[str]]:
    """Normalize one or repeated --model-path uses into model identifiers."""
    if not model_paths:
        return [None]

    flattened: List[str] = []
    for item in model_paths:
        if isinstance(item, (list, tuple)):
            flattened.extend(str(x) for x in item)
        else:
            flattened.append(str(item))

    cleaned = [mp.strip() for mp in flattened if mp and mp.strip()]
    return cleaned or [None]


def _generation_metadata(args, model_paths: List[Optional[str]]) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_paths": model_paths,
        "context_mode": args.context,
        "had_initial_context": bool(getattr(args, "_had_initial_context", False)),
        "context_reset_occurred": any(t.get("context_reset") for t in transcript_turns),
        "temperature": 0,
        "deterministic": True,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt_provided": bool(args.system),
        "system_prompt": args.system if args.system else "",
        "batch_mode": bool(args.batch_file),
        "batch_file": args.batch_file or "",
        "session_file": args.session_file or "",
        "cot": bool(args.cot),
    }


def save_transcript(args=None, model_paths: Optional[List[Optional[str]]] = None):
    if not transcript_turns:
        return
    try:
        filename = input("[~] Enter filename to save transcript: ").strip()
        if not filename:
            print("[-] No filename entered. Transcript not saved.")
            return
        if not filename.lower().endswith(".json"):
            filename += ".json"

        payload = {
            "metadata": _generation_metadata(args, model_paths or []) if args is not None else {},
            "turns": transcript_turns,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[+] Transcript saved to {filename}")
    except Exception as e:
        print(f"[-] Failed to save transcript: {e}")


def load_session_history(session_file: str) -> List[Dict[str, str]]:
    if not session_file or not os.path.exists(session_file):
        return []
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cleaned = []
            for m in data:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    cleaned.append({"role": str(m["role"]), "content": str(m["content"])})
            return cleaned
    except Exception as e:
        print(f"[-] Could not load session history from {session_file}: {e}")
    return []


def save_session_history(session_file: str, history: List[Dict[str, str]]) -> None:
    if not session_file:
        return
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[~] Session saved to {session_file}")
    except Exception as e:
        print(f"[-] Failed to save session to {session_file}: {e}")


def load_text_file(path_str: str) -> str:
    """Load UTF-8 text from a file path."""
    if not path_str:
        return ""
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read text file {path_str}: {e}") from e


def resolve_system_prompt(args) -> str:
    """Resolve the system instruction from --system and/or --system-file."""
    inline = (getattr(args, "system", "") or "").strip()
    file_path = (getattr(args, "system_file", "") or "").strip()

    if inline and file_path:
        raise RuntimeError("Use only one of --system or --system-file, not both.")

    if file_path:
        return load_text_file(file_path).strip()

    return inline


def load_batch_prompts(batch_file: str, skip_empty: bool = True) -> List[str]:
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    prompts: List[str] = []
    with open(batch_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            prompt = line.strip()
            if skip_empty and not prompt.strip():
                continue
            prompts.append(prompt)

    return prompts


def save_batch_results(output_file: str, rows: List[Dict[str, Any]]) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[+] Batch results saved to {output_file}")


def parse_batch_indices(batch_index: str, total_prompts: int) -> List[int]:
    """Parse a comma-delimited list of 1-based batch prompt indices."""
    if not batch_index or not batch_index.strip():
        return []

    selected: List[int] = []
    seen = set()

    for raw_part in batch_index.split(','):
        part = raw_part.strip()
        if not part:
            raise ValueError("--batch-index contains an empty index; use comma-delimited integers like 1,3,5.")

        try:
            index = int(part)
        except ValueError as e:
            raise ValueError(f"--batch-index must contain only integers; got {part!r}.") from e

        if index < 1:
            raise ValueError(f"--batch-index values start at 1; got {index}.")
        if index > total_prompts:
            raise ValueError(
                f"--batch-index value {index} is out of range; batch file has {total_prompts} prompts."
            )

        if index not in seen:
            selected.append(index)
            seen.add(index)

    return selected


# -------------------------
# Prompt builders (local - non gpt-oss)
# -------------------------

def build_prompt_from_messages(
    tokenizer,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str
) -> str:
    """
    Build a chat prompt using the tokenizer's own chat template when available.
    Falls back to a simple USER/ASSISTANT format only for tokenizers without a chat template.
    """
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_input})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: List[str] = []
    for msg in msgs:
        role = (msg.get("role") or "user").strip().upper()
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


# -------------------------
# gpt-oss Harmony helpers (local)
# -------------------------

def _load_harmony():
    """
    Lazy import so non-gpt-oss setups don't require openai_harmony installed.
    """
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding, Conversation, Message, Role, SystemContent, DeveloperContent


def _build_harmony_conversation(
    *,
    Conversation: Any,
    Message: Any,
    Role: Any,
    SystemContent: Any,
    DeveloperContent: Any,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
) -> Any:
    """
    Build a Harmony Conversation for gpt-oss.
    """
    msgs = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]
    if system_prompt and system_prompt.strip():
        msgs.append(
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(system_prompt.strip())
            )
        )

    for m in history:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            msgs.append(Message.from_role_and_content(Role.ASSISTANT, content))
        else:
            msgs.append(Message.from_role_and_content(Role.USER, content))

    msgs.append(Message.from_role_and_content(Role.USER, user_input))
    return Conversation.from_messages(msgs)


def _extract_harmony_text(entries: List[Any], *, show_cot: bool) -> str:
    final_parts: List[str] = []
    cot_parts: List[str] = []

    for msg in entries or []:
        try:
            d = msg.to_dict()
        except Exception:
            continue

        channel = d.get("channel")
        content = d.get("content")
        text_chunks: List[str] = []

        if isinstance(content, str):
            text_chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        text_chunks.append(t)

        text = "\n".join(t.strip() for t in text_chunks if t and t.strip()).strip()
        if not text:
            continue

        if channel in ("analysis", "reasoning", "thought"):
            cot_parts.append(text)
        else:
            final_parts.append(text)

    if show_cot and cot_parts:
        return (
            "=== Chain of Thought ===\n"
            + "\n\n".join(cot_parts)
            + "\n\n=== Final Answer ===\n"
            + "\n".join(final_parts)
        ).strip()

    return "\n".join(final_parts).strip()


def _fallback_strip_harmony_decode(decoded: str) -> str:
    if not isinstance(decoded, str):
        return ""
    marker = "<|channel|>final<|message|>"
    if marker in decoded:
        return decoded.split(marker)[-1].strip()
    return decoded.strip()


# -------------------------
# Local generation
# -------------------------

def generate_with_prompt(
    model,
    tokenizer,
    device,
    full_prompt: str,
    max_new_tokens: int = 256
) -> str:
    tgt_device = torch.device(device) if isinstance(device, str) else device
    inputs = tokenizer([full_prompt], return_tensors="pt")
    inputs = {k: v.to(tgt_device) for k, v in inputs.items()}

    gen_kwargs = dict(
        do_sample=False,
        num_beams=1,
        min_new_tokens=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_gpt_oss_with_harmony(
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
    show_cot: bool,
) -> str:
    encoding, Conversation, Message, Role, SystemContent, DeveloperContent = _load_harmony()

    convo = _build_harmony_conversation(
        Conversation=Conversation,
        Message=Message,
        Role=Role,
        SystemContent=SystemContent,
        DeveloperContent=DeveloperContent,
        system_prompt=system_prompt,
        history=history,
        user_input=user_input,
    )

    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    input_ids = torch.tensor([prefill_ids], dtype=torch.long, device=model.device)

    gen_kwargs = dict(
        do_sample=False,
        num_beams=1,
        min_new_tokens=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_token_ids,
    )

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, **gen_kwargs)

    full = out[0].tolist()
    completion_ids = full[len(prefill_ids):]

    try:
        entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
        extracted = _extract_harmony_text(entries, show_cot=show_cot)
        if extracted:
            return extracted
    except Exception:
        pass

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    return _fallback_strip_harmony_decode(decoded)


# -------------------------
# API backends (no branching, temperature=0)
# -------------------------

def openai_generate_once(
    model_id: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_input})

    payload = {
        "model": model_id,
        "input": msgs,
        "temperature": 0,
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

    data = r.json()

    parts: List[str] = []
    for item in data.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    parts.append(c["text"])

    if not parts and isinstance(data.get("output_text"), str):
        parts.append(data["output_text"])

    return ("\n".join(parts)).strip()


def _vertex_access_token() -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception as e:
        raise RuntimeError("google-auth is required for Vertex calls (pip install google-auth).") from e

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    if not getattr(creds, "token", None):
        raise RuntimeError("Could not obtain Google access token via ADC")
    return str(creds.token)


def vertex_generate_once(
    endpoint_resource: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
) -> str:
    m = re.match(r"^projects/(\d+)/locations/([^/]+)/endpoints/(\d+)$", endpoint_resource)
    if not m:
        raise ValueError(f"Not a valid Vertex endpoint resource: {endpoint_resource}")
    location = m.group(2)

    token = _vertex_access_token()
    url = f"https://{location}-aiplatform.googleapis.com/v1/{endpoint_resource}:generateContent"

    contents: List[Dict[str, object]] = []
    for msg in history:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            role = "model"
        elif role not in ("user", "model"):
            role = "user"
        contents.append({"role": role, "parts": [{"text": content}]})

    contents.append({"role": "user", "parts": [{"text": user_input}]})

    payload: Dict[str, object] = {
        "contents": contents,
        "generationConfig": {"temperature": 0},
    }

    if system_prompt and system_prompt.strip():
        payload["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Vertex API error {r.status_code}: {r.text}")

    data = r.json()

    candidates = data.get("candidates") or []
    if isinstance(candidates, list) and candidates:
        c0 = candidates[0] if isinstance(candidates[0], dict) else None
        if c0:
            content = c0.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts = []
                    for p in parts:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            texts.append(p["text"])
                    if texts:
                        return "".join(texts).strip()

    return json.dumps(data, ensure_ascii=False)[:2000]


# -------------------------
# Unified single-turn runner
# -------------------------

def run_single_prompt(
    *,
    backend: str,
    args,
    model,
    tokenizer,
    device,
    is_gpt_oss_local: bool,
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    history = history or []

    if backend == "local":
        if is_gpt_oss_local:
            return generate_gpt_oss_with_harmony(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                system_prompt=args.system,
                history=history,
                user_input=prompt,
                show_cot=args.cot,
            )

        full_prompt = build_prompt_from_messages(
            tokenizer=tokenizer,
            system_prompt=args.system,
            history=history,
            user_input=prompt,
        )

        return generate_with_prompt(
            model, tokenizer, device, full_prompt, max_new_tokens=args.max_new_tokens
        )

    if backend == "openai":
        return openai_generate_once(
            model_id=args.model_path,
            system_prompt=args.system,
            history=history,
            user_input=prompt,
        )

    return vertex_generate_once(
        endpoint_resource=args.model_path,
        system_prompt=args.system,
        history=history,
        user_input=prompt,
    )



def _load_model_spec_in_place(spec: Dict[str, Any], args) -> None:
    """Load a local model once and cache it on its model spec. API specs are left unloaded."""
    if spec.get("backend") != "local" or spec.get("model") is not None:
        return

    model, tokenizer, device, resolved = load_local_model_and_tokenizer(
        spec["model_path"],
        base_model_path=args.base_model_path,
    )
    name_or_path = getattr(tokenizer, "name_or_path", None)
    is_gpt_oss_local = is_gpt_oss_model_name(str(resolved)) or is_gpt_oss_model_name(name_or_path)
    spec.update({
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "resolved": resolved,
        "is_gpt_oss_local": is_gpt_oss_local,
    })
    print(f"[~] Loaded model from {resolved} onto {device}")


def _run_for_model_spec(
    *,
    spec: Dict[str, Any],
    args,
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Run one prompt for one model spec. Local models are loaded once at startup and reused."""
    local_args = deepcopy(args)
    local_args.model_path = spec["model_path"]

    if spec["backend"] == "local" and spec.get("model") is None:
        raise RuntimeError(
            f"Local model {spec['model_path']!r} was not loaded before inference."
        )

    return run_single_prompt(
        backend=spec["backend"],
        args=local_args,
        model=spec.get("model"),
        tokenizer=spec.get("tokenizer"),
        device=spec.get("device"),
        is_gpt_oss_local=bool(spec.get("is_gpt_oss_local", False)),
        prompt=prompt,
        history=history or [],
    )

# -------------------------
# Batch execution
# -------------------------

def run_batch_mode(
    *,
    model_specs: List[Dict[str, Any]],
    args,
) -> int:
    prompts = load_batch_prompts(args.batch_file, skip_empty=(not args.batch_keep_empty))
    print(f"[~] Loaded {len(prompts)} prompts from {args.batch_file}")

    if not prompts:
        print("[-] No prompts found in batch file.")
        return 1

    selected_indices = parse_batch_indices(args.batch_index, len(prompts))
    if selected_indices:
        selected_set = set(selected_indices)
        indexed_prompts = [(idx, prompt) for idx, prompt in enumerate(prompts, start=1) if idx in selected_set]
        print(f"[~] Running {len(indexed_prompts)} selected batch prompts: {','.join(map(str, selected_indices))}")
    else:
        indexed_prompts = list(enumerate(prompts, start=1))

    results: List[Dict[str, Any]] = []

    output_f = None
    if args.batch_output:
        output_f = open(args.batch_output, "w", encoding="utf-8")

    try:
        for idx, prompt in tqdm(
            indexed_prompts,
            desc="Batch prompts",
            unit="prompt",
            dynamic_ncols=True,
        ):
            try:
                per_model_results = []
                for spec in model_specs:
                    try:
                        reply = _run_for_model_spec(
                            spec=spec,
                            args=args,
                            prompt=prompt,
                            history=[],
                        )
                        status = "ok"
                        error = None
                    except Exception as e:
                        reply = ""
                        status = "error"
                        error = str(e)
                        tqdm.write(f"[-] Batch prompt {idx} failed for {spec['model_path']}: {e}")

                    result = {
                        "index": idx,
                        "prompt": prompt,
                        "model": spec["model_path"],
                        "response": reply,
                        "status": status,
                    }
                    if error is not None:
                        result["error"] = error
                    per_model_results.append(result)

                    if output_f is not None:
                        output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_f.flush()
                        os.fsync(output_f.fileno())
                    else:
                        results.append(result)

                    transcript_turns.append({
                        "turn_index": idx,
                        "mode": "batch",
                        "input": prompt,
                        "output": reply,
                        "model": spec["model_path"],
                        "status": status,
                        **({"error": error} if error is not None else {}),
                    })
            except Exception as e:
                tqdm.write(f"[-] Batch prompt {idx} failed unexpectedly: {e}")

    finally:
        if output_f is not None:
            output_f.close()

    if args.batch_output:
        print(f"[+] Batch results saved incrementally to {args.batch_output}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


# -------------------------
# Branch preview utilities (local only; non-gpt-oss)
# -------------------------

@torch.inference_mode()
def _beam_start_candidates(
    model,
    tokenizer,
    device,
    full_prompt: str,
    k: int = 5,
    depth: int = 6,
) -> List[Tuple[str, float]]:
    if k <= 0 or depth <= 0:
        return []

    inputs = tokenizer(full_prompt, return_tensors="pt")
    for key in list(inputs.keys()):
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)
    input_ids = inputs["input_ids"]

    beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]

    for _ in range(depth):
        new_beams: List[Tuple[torch.Tensor, float]] = []
        for ids, score in beams:
            outputs = model(input_ids=ids)
            logits = outputs.logits[:, -1, :]
            logprobs = F.log_softmax(logits, dim=-1)

            topk_vals, topk_idx = torch.topk(logprobs, k)
            for j in range(topk_idx.size(-1)):
                tok_id = topk_idx[0, j].view(1, 1)
                new_ids = torch.cat([ids, tok_id], dim=-1)
                new_score = score + float(topk_vals[0, j])
                new_beams.append((new_ids, new_score))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:k]

        finished = 0
        for ids, _ in beams:
            text = tokenizer.decode(ids[0][input_ids.size(-1):], skip_special_tokens=True)
            if "\n" in text:
                finished += 1
        if finished >= max(1, k // 2):
            break

    results_text_scores: List[Tuple[str, float]] = []
    seen = set()
    for ids, score in beams:
        cont = tokenizer.decode(ids[0][input_ids.size(-1):], skip_special_tokens=True)
        first_line = cont.splitlines()[0].strip()
        if first_line and first_line not in seen:
            results_text_scores.append((first_line, score))
            seen.add(first_line)

    if not results_text_scores:
        return []
    scores = torch.tensor([s for _, s in results_text_scores], dtype=torch.float32)
    m = float(torch.max(scores))
    probs = torch.exp(scores - (m + float(torch.log(torch.sum(torch.exp(scores - m))))))
    results = [(txt, float(p)) for (txt, _), p in zip(results_text_scores, probs)]
    return results[:k]


@torch.inference_mode()
def _greedy_start_candidate(
    model,
    tokenizer,
    device,
    full_prompt: str,
    depth: int = 6,
) -> str:
    if depth <= 0:
        return ""

    inputs = tokenizer(full_prompt, return_tensors="pt")
    for key in list(inputs.keys()):
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)

    ids = inputs["input_ids"]
    start_len = ids.size(-1)

    for _ in range(depth):
        outputs = model(input_ids=ids)
        logits = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1).view(1, 1)
        ids = torch.cat([ids, next_id], dim=-1)

        seg = tokenizer.decode(ids[0][start_len:], skip_special_tokens=True)
        if "\n" in seg:
            break

    cont = tokenizer.decode(ids[0][start_len:], skip_special_tokens=True)
    return cont.splitlines()[0].strip()


class BranchingCompleter(Completer):
    def __init__(
        self,
        model,
        tokenizer,
        device,
        args,
        history_ref: List[Dict[str, str]],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.history_ref = history_ref

    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        try:
            user_partial = document.text_before_cursor

            prompt_for_model = build_prompt_from_messages(
                tokenizer=self.tokenizer,
                system_prompt=self.args.system,
                history=(self.history_ref if self.args.context == "session" else []),
                user_input=user_partial,
            )

            greedy_txt = _greedy_start_candidate(
                self.model, self.tokenizer, self.device,
                full_prompt=prompt_for_model,
                depth=self.args.branch_depth,
            )
            if greedy_txt:
                display = f"{greedy_txt:<45}  *"
                yield Completion(text="", start_position=0, display=display, display_meta="greedy")

            want_alternatives = self.args.branch_k if self.args.branch else 0
            if want_alternatives > 0:
                oversample_k = max(want_alternatives * 2, want_alternatives + 3)

                beams_raw = _beam_start_candidates(
                    self.model, self.tokenizer, self.device,
                    full_prompt=prompt_for_model,
                    k=oversample_k,
                    depth=self.args.branch_depth,
                )

                alts: List[Tuple[str, float]] = []
                seen = set([greedy_txt] if greedy_txt else [])
                for b_txt, b_prob in beams_raw:
                    if not b_txt or b_txt in seen:
                        continue
                    seen.add(b_txt)
                    alts.append((b_txt, b_prob))
                    if len(alts) >= want_alternatives:
                        break

                for b_txt, b_prob in alts[:want_alternatives]:
                    pct = f"{b_prob * 100:.1f}%"
                    display = f"{b_txt:<45}  {pct}"
                    yield Completion(text="", start_position=0, display=display, display_meta="model branch")
        except Exception:
            return

def load_causal_lm_bf16(model_path: str, **kwargs):
    """Load a causal LM with dtype=bf16 on newer Transformers, falling back for older versions."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )


def make_generation_deterministic(model, tokenizer) -> None:
    """Ensure saved generation_config values do not re-enable sampling or invalid flags."""
    if hasattr(model, "config"):
        model.config.use_cache = True
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, "generation_config"):
        gc = model.generation_config
        gc.do_sample = False
        gc.num_beams = 1
        gc.pad_token_id = tokenizer.pad_token_id
        gc.eos_token_id = tokenizer.eos_token_id

        for name in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gc, name):
                setattr(gc, name, None)


def load_local_model_and_tokenizer(model_path: str, base_model_path: Optional[str] = None):
    """
    Load either:
      - a normal full HF model, or
      - a PEFT/LoRA adapter directory with --base-model-path.
    """
    if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        if not _HAVE_PEFT:
            raise RuntimeError("PEFT adapter detected, but peft is not installed. Run: pip install peft")

        adapter_path = model_path

        if base_model_path:
            resolved_base = base_model_path
        else:
            cfg = PeftConfig.from_pretrained(adapter_path)
            resolved_base = cfg.base_model_name_or_path

        print(f"[~] PEFT adapter detected: {adapter_path}")
        print(f"[~] Loading base model: {resolved_base}")

        tokenizer = AutoTokenizer.from_pretrained(resolved_base, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = load_causal_lm_bf16(
            resolved_base,
            device_map={"": 0},
            trust_remote_code=False,
        )

        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        make_generation_deterministic(model, tokenizer)

        device = next(model.parameters()).device
        return model, tokenizer, device, adapter_path

    model, tokenizer, device, resolved = load_model_and_tokenizer(model_path)
    make_generation_deterministic(model, tokenizer)
    return model, tokenizer, device, resolved

def main():
    parser = argparse.ArgumentParser(description="Interactively prompt a model")
    parser.add_argument("--model-path", nargs="+", action="append", default=None, help="One or more model directories, HuggingFace repository IDs, OpenAI fine-tunes, or Vertex endpoints; may be repeated")

    parser.add_argument(
        "--context",
        choices=["none", "session"],
        default="none",
        help="Conversation context mode: 'none' (stateless, default) or 'session' (persist history to a file)."
    )
    parser.add_argument(
        "--session-file",
        type=str,
        default="",
        help="JSON file to load/save persistent session history (used when --context=session)."
    )
    parser.add_argument("--system", type=str, default="", help="Optional inline system prompt to steer the model.")
    parser.add_argument(
        "--system-file",
        type=str,
        default="",
        help="Optional path to a UTF-8 text file containing the system instruction to apply across all backends."
    )

    parser.add_argument(
        "--cot",
        action="store_true",
        help="Show chain-of-thought / internal reasoning (gpt-oss local only).",
    )

    parser.add_argument(
        "--branch",
        action="store_true",
        help="Also show likely alternative openings (beam search) while typing. Greedy preview is always shown (local models only; disabled for gpt-oss)."
    )
    parser.add_argument(
        "--branch-k",
        type=int,
        default=3,
        help="Number of alternative openings to preview when --branch is enabled (greedy '*' is shown in addition)."
    )
    parser.add_argument(
        "--branch-depth",
        type=int,
        default=8,
        help="Tokens per preview to look ahead for both greedy and beam when previewing."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per turn."
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default="",
        help="Optional path to a plain text file containing one prompt per line. Each line runs as its own fresh context."
    )
    parser.add_argument(
        "--batch-output",
        type=str,
        default="",
        help="Optional path to write batch results as JSON. If omitted, results are printed to stdout."
    )
    parser.add_argument(
        "--batch-keep-empty",
        action="store_true",
        help="Process empty lines in the batch file instead of skipping them."
    )
    parser.add_argument(
        "--batch-index",
        type=str,
        default="",
        help=(
            "Optional comma-delimited 1-based batch prompt indices to run, "
            "for example: 1,3,5. Original prompt indices are preserved in output."
        ),
    )

    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Base HF model ID/path to use when --model-path points to a PEFT/LoRA adapter directory.",
    )

    args = parser.parse_args()

    try:
        args.system = resolve_system_prompt(args)
    except Exception as e:
        parser.error(str(e))

    if args.branch_k < 1:
        parser.error("--branch-k must be at least 1")

    if args.batch_index and not args.batch_file:
        parser.error("--batch-index requires --batch-file")

    if args.batch_file and args.context == "session":
        print("[~] Batch mode ignores session context; each prompt runs independently.")

    model_paths = _split_model_paths(args.model_path)
    model_specs: List[Dict[str, Any]] = []
    for mp in model_paths:
        backend = detect_backend(mp)
        model_specs.append({"model_path": mp, "backend": backend})

    for backend_name in sorted({spec["backend"] for spec in model_specs}):
        if backend_name in ("openai", "vertex"):
            try:
                ensure_backend_credentials(backend_name)
            except Exception as e:
                parser.error(str(e))

    if any(spec["backend"] in ("openai", "vertex") for spec in model_specs):
        args.branch = False

    if len(model_specs) > 1 and args.branch:
        args.branch = False
        print("[~] Multiple models detected: disabling live preview/branching.")

    model = tokenizer = device = resolved = None
    is_gpt_oss_local = False

    first_spec = model_specs[0]
    for spec in model_specs:
        if spec["backend"] == "local":
            _load_model_spec_in_place(spec, args)
        else:
            print(f"[~] Queued {spec['backend']} model/endpoint: {spec['model_path']}")

    if len(model_specs) == 1 and first_spec["backend"] == "local":
        args.model_path = first_spec["model_path"]
        model = first_spec.get("model")
        tokenizer = first_spec.get("tokenizer")
        device = first_spec.get("device")
        resolved = first_spec.get("resolved")
        is_gpt_oss_local = bool(first_spec.get("is_gpt_oss_local", False))

        if is_gpt_oss_local and args.branch:
            args.branch = False
            print("[~] gpt-oss detected: disabling live preview/branching (Harmony prompt/parse).")

        if args.cot and not is_gpt_oss_local:
            print("[~] Note: --cot only affects local gpt-oss (ignored for this model).")
    else:
        if any(spec["backend"] != "local" for spec in model_specs):
            print("[~] Live preview and branching disabled to avoid API usage while typing.")
        if args.cot:
            print("[~] Note: --cot only affects local gpt-oss models.")

    if args.batch_file:
        exit_code = run_batch_mode(
            model_specs=model_specs,
            args=args,
        )
        save_transcript(args, model_paths)
        raise SystemExit(exit_code)

    global messages
    if args.context == "session":
        prior = load_session_history(args.session_file)
        messages = prior
        args._had_initial_context = bool(prior)
        if prior:
            print(f"[~] Loaded {len(prior)} prior messages from {args.session_file}.")
    else:
        messages = []
        args._had_initial_context = False

    prompt_session = None
    if _USE_PT:
        history_obj = FileHistory(".prompt_history.txt") if args.context == "session" else InMemoryHistory()
        completer = None
        complete_while_typing = False

        if len(model_specs) == 1 and first_spec["backend"] == "local" and args.branch and not is_gpt_oss_local:
            completer = BranchingCompleter(
                model=model,
                tokenizer=tokenizer,
                device=device,
                args=args,
                history_ref=messages,
            )
            complete_while_typing = True
            print("[~] Preview enabled: greedy '*' plus requested alternatives (local only).")
        else:
            if len(model_specs) == 1 and first_spec["backend"] == "local":
                if is_gpt_oss_local:
                    print("[~] Preview disabled for gpt-oss.")
                else:
                    print("[~] Preview disabled (run with --branch to enable for local models).")

        prompt_session = PromptSession(
            history=history_obj,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=complete_while_typing,
        )

        if complete_while_typing:
            try:
                buf = prompt_session.default_buffer

                def _trigger_completion(_event=None):
                    if buf.text:
                        buf.start_completion(select_first=False)

                buf.on_text_changed += _trigger_completion
            except Exception:
                pass

        print("[+] Model ready. Type your prompt. Press Ctrl+D to exit. Ctrl+C cancels current entry.\n")
        print("[~] Tip: type /reset to clear the conversation context.\n")
    elif _USE_RL:
        hist_path = ".readline_history"
        try:
            if os.path.exists(hist_path):
                readline.read_history_file(hist_path)  # type: ignore
        except Exception:
            pass
        print("[+] Model ready (readline). Type your prompt. Press Ctrl+D to exit. Ctrl+C cancels current entry.\n")
        print("[~] Tip: type /reset to clear the conversation context.\n")
    else:
        print("[+] Model ready (basic input). Type your prompt. Press Ctrl+D to exit. Ctrl+C cancels current entry.\n")
        print("[~] Tip: type /reset to clear the conversation context.\n")

    user_turn_index = 0

    try:
        while True:
            try:
                if _USE_PT and prompt_session is not None:
                    prompt = prompt_session.prompt(">>> ")
                else:
                    prompt = input(">>> ")
            except KeyboardInterrupt:
                print("^C")
                continue
            except EOFError:
                print("\n[~] Exiting...")
                break

            cmd = prompt.strip()
            if not cmd:
                continue

            if cmd == "/reset":
                user_turn_index += 1
                messages.clear()
                if args.context == "session":
                    save_session_history(args.session_file, messages)
                transcript_turns.append({
                    "turn_index": user_turn_index,
                    "input": prompt,
                    "output": "Context reset: cleared conversation history.",
                    "model": None,
                    "context_reset": True,
                })
                print("[~] Context reset: cleared conversation history.")
                continue

            user_turn_index += 1
            replies_for_history: List[Dict[str, str]] = []
            for spec in model_specs:
                try:
                    reply = _run_for_model_spec(
                        spec=spec,
                        args=args,
                        prompt=prompt,
                        history=(messages if args.context == "session" else []),
                    )
                    status = "ok"
                    error = None
                except KeyboardInterrupt:
                    print("^C")
                    status = "cancelled"
                    reply = ""
                    error = "KeyboardInterrupt"
                except Exception as e:
                    status = "error"
                    reply = ""
                    error = str(e)

                if len(model_specs) > 1:
                    print(f"\n=== {spec['model_path']} ===")
                if reply:
                    print(reply)
                elif error:
                    print(f"[-] {error}")

                transcript_turns.append({
                    "turn_index": user_turn_index,
                    "input": prompt,
                    "output": reply,
                    "model": spec["model_path"],
                    "context_reset": False,
                    "status": status,
                    **({"error": error} if error is not None else {}),
                })
                if status == "ok":
                    replies_for_history.append({"model": spec["model_path"], "reply": reply})

            if args.context == "session":
                messages.append({"role": "user", "content": prompt})
                if len(replies_for_history) == 1:
                    messages.append({"role": "assistant", "content": replies_for_history[0]["reply"]})
                elif replies_for_history:
                    combined = "\n\n".join(
                        f"[{r['model']}]\n{r['reply']}" for r in replies_for_history
                    )
                    messages.append({"role": "assistant", "content": combined})

    except KeyboardInterrupt:
        print("^C")
    finally:
        if _USE_RL:
            try:
                readline.write_history_file(".readline_history")  # type: ignore
            except Exception:
                pass

        if args.context == "session":
            save_session_history(args.session_file, messages)

    save_transcript(args, model_paths)


if __name__ == "__main__":
    main()
