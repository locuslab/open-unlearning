from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

# Try to import diffusion adapter from main repo (if available)
try:
    from dllm.integrations.open_unlearning_adapter import wrap_model_if_diffusion
    _DIFFUSION_ADAPTER_AVAILABLE = True
except ImportError:
    _DIFFUSION_ADAPTER_AVAILABLE = False
    def wrap_model_if_diffusion(model, tokenizer, config=None):
        """Fallback if adapter not available."""
        return model

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None, ValueError("Model config not found.")
    with open_dict(model_cfg):
        model_args_dict = model_cfg.get("model_args", None)
        assert model_args_dict is not None, ValueError("model_args absent in configs/model.")
        tokenizer_args = model_cfg.get("tokenizer_args", None)
        model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    # Convert to regular dict to ensure proper access
    model_args = OmegaConf.to_container(model_args_dict, resolve=True) if isinstance(model_args_dict, DictConfig) else model_args_dict
    torch_dtype = get_dtype(model_args)
    model_cls = MODEL_REGISTRY[model_handler]
    model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    
    # Auto-wrap diffusion models to be compatible with AR-based metrics
    # (only if adapter is available from main dllm repo)
    if _DIFFUSION_ADAPTER_AVAILABLE:
        diffusion_config = model_cfg.get("diffusion_adapter", None)
        model = wrap_model_if_diffusion(model, tokenizer, config=diffusion_config)
    
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
