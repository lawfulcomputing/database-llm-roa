"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.2.4
Description: Common functions for inference scripts.
"""

import os, re
from typing import Optional, Tuple, List, Union, Dict
import torch

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)

try:
    from transformers import BitsAndBytesConfig
    _HAVE_BNB = True
except Exception:
    _HAVE_BNB = False


def load_model_and_tokenizer(model_path: Optional[str]) -> Tuple[object, object, str, str]:
    """
    Load a standard HF CausalLM model/tokenizer STRICTLY on a single device.
    Returns (model, tokenizer, device, resolved_path).

    Env toggles (optional):
      - HF_LOAD_IN_4BIT=1  -> load with 4-bit NF4 (bitsandbytes)
      - HF_LOAD_IN_8BIT=1  -> load with 8-bit (bitsandbytes)
    """
    resolved = resolve_model_source(model_path)
    device = _single_device()  # e.g., 'cuda:0' or 'cpu'

    device_map = {"": 0} if device.startswith("cuda") else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    quantization_config = _bnb_config_from_env()
    trust_remote_code = _env_flag("HF_TRUST_REMOTE_CODE")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(
        resolved,
        trust_remote_code=trust_remote_code,
    )

    if hasattr(config, "quantization_config") and config.quantization_config is None:
        # Best option: delete the attribute so to_dict() won't call .to_dict() on None
        try:
            delattr(config, "quantization_config")
        except Exception:
            # Fallback: replace with a harmless object that has to_dict()
            class _SafeQuantConfig:
                def to_dict(self):
                    return {}
            config.quantization_config = _SafeQuantConfig()

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        config=config,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
    )

    if device.startswith("cuda"):
        try:
            model.to(device)
        except Exception:
            pass

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Conservative default generation config
    try:
        gc = GenerationConfig.from_model_config(model.config)
        gc.do_sample = False
        gc.num_beams = 1
        gc.min_new_tokens = 1
        existing_max = getattr(getattr(model, "generation_config", None), "max_new_tokens", None)
        gc.max_new_tokens = existing_max or 64
        gc.pad_token_id = tokenizer.pad_token_id
        gc.eos_token_id = tokenizer.eos_token_id
        model.generation_config = gc
    except Exception:
        pass

    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return model, tokenizer, device, resolved

def resolve_model_source(path: Optional[str]) -> str:
    """
    Returns a local directory path or an HF repo id.
    - If `path` is provided: resolve as repo id or local (base-dir > newest checkpoint).
    - If `path` is None: auto-discover shallowest model dir under CWD (no name assumptions).
    """
    if path:
        if path.startswith("hf://"):
            return path[5:]
        if _looks_like_repo_id(path):
            return path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path does not exist: {path}")

        if os.path.isdir(path):
            if _is_hf_model_dir(path) or _is_peft_dir(path) or _dir_has_model_markers(path):
                return path
            ckpts = _list_checkpoints(path)
            if ckpts:
                return ckpts[0]
            raise FileNotFoundError(
                f"No model markers in {path} and no checkpoint-* subdirs found."
            )

        # file path; let from_pretrained validate
        return path

    # Auto-discover under CWD
    auto = _auto_discover_model_dir(".")
    if auto:
        return auto
    raise FileNotFoundError(
        "Could not auto-discover a model directory. "
        "No path was provided and no directory with model markers (e.g., config.json) was found under the current directory."
    )

def _looks_like_repo_id(s: str) -> bool:
    if s.startswith("hf://"):
        s = s[5:]
    if os.path.isabs(s) or os.path.exists(s):
        return False
    return "/" in s  # "org/name" (optionally "@rev")

def _single_device() -> str:
    """Return a concrete single-device string: 'cuda:0' if available else 'cpu'."""
    if torch.cuda.is_available():
        # Hard-pin to the first visible GPU. You can change to a specific index if you like.
        torch.cuda.set_device(0)
        return "cuda:0"
    return "cpu"

def _bnb_config_from_env():
    """Create a BitsAndBytesConfig if env flags request it; else return None."""
    load_in_4bit = _env_flag("HF_LOAD_IN_4BIT")
    load_in_8bit = _env_flag("HF_LOAD_IN_8BIT")
    if load_in_4bit and load_in_8bit:
        raise ValueError("Set only one of HF_LOAD_IN_4BIT or HF_LOAD_IN_8BIT.")
    if not (load_in_4bit or load_in_8bit):
        return None
    if not _HAVE_BNB:
        raise RuntimeError("bitsandbytes requested via env, but transformers BitsAndBytesConfig not available.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)

def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y"}