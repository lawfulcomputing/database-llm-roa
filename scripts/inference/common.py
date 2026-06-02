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
