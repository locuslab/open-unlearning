from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

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
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        # Note: we deliberately do NOT pass cache_dir here. Transformers resolves
        # the cache from the HF_HOME / HF_HUB_CACHE env vars automatically, placing
        # models under $HF_HOME/hub (the HF convention). Passing cache_dir=$HF_HOME
        # would instead drop models directly under $HF_HOME (no /hub subdir) and
        # diverge from where `datasets` caches data.
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            dtype=torch_dtype,  # transformers>=4.56 renamed `torch_dtype` -> `dtype`
            **model_args,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    # Optional: wrap the loaded base model with a PEFT/LoRA adapter.
    # This is additive and only triggers when a `peft_args` block is present in
    # the model config, so existing (non-LoRA) configs are unaffected.
    peft_args = model_cfg.get("peft_args", None)
    if peft_args is not None:
        model = get_peft_lora_model(model, peft_args)

    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def get_peft_lora_model(model, peft_args: DictConfig):
    """Wrap a base causal LM with a LoRA adapter using the `peft` library.

    Args:
        model: a freshly loaded base model (e.g. AutoModelForCausalLM).
        peft_args (DictConfig): LoRA hyper-parameters. Recognised keys mirror
            `peft.LoraConfig` (r, lora_alpha, lora_dropout, target_modules,
            bias, task_type, ...). An optional `path` key can point to an
            existing adapter checkpoint to resume/evaluate instead of creating
            a fresh adapter.
    """
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError as e:
        raise ImportError(
            "LoRA finetuning requires the `peft` library. Install it with "
            "`pip install peft`."
        ) from e

    with open_dict(peft_args):
        adapter_path = peft_args.pop("path", None)

    if adapter_path is not None:
        # Load a previously trained LoRA adapter on top of the base model.
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        logger.info(f"Loaded existing LoRA adapter from {adapter_path}")
    else:
        lora_config = LoraConfig(**peft_args)
        model = get_peft_model(model, lora_config)
        logger.info("Created a new LoRA adapter on top of the base model.")
    model.print_trainable_parameters()
    return model


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
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg)
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
