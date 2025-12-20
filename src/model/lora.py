from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import DictConfig, open_dict, ListConfig
from typing import Optional
import torch
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment variables or .env file.
    Reads HF_TOKEN variable.
    """
    token = os.getenv("HF_TOKEN", default=None)
    return token


class LoRAModelForCausalLM:
    """
    Wrapper class for loading models with LoRA adapters.
    Supports the specified LoRA configuration parameters.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        lora_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        """
        Load a model with LoRA adapters.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            lora_config: LoRA configuration parameters
            **kwargs: Additional arguments for model loading
        """
        # Default LoRA configuration
        default_lora_config = {
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
                "lm_head",
            ],
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "r": 128,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Merge with provided config
        if lora_config:
            lora_params = dict(lora_config)
        else:
            lora_params = default_lora_config.copy()

        # Convert OmegaConf objects to regular Python types for JSON serialization
        def convert_omegaconf_to_python(obj):
            """Convert OmegaConf objects to regular Python types."""
            if isinstance(obj, ListConfig):
                return [convert_omegaconf_to_python(item) for item in obj]
            elif isinstance(obj, DictConfig):
                return {k: convert_omegaconf_to_python(v) for k, v in obj.items()}
            elif hasattr(obj, "_content"):  # Fallback for other OmegaConf types
                if isinstance(obj._content, list):
                    return [convert_omegaconf_to_python(item) for item in obj._content]
                elif isinstance(obj._content, dict):
                    return {
                        k: convert_omegaconf_to_python(v)
                        for k, v in obj._content.items()
                    }
                else:
                    return obj._content
            else:
                return obj

        # Convert all parameters to ensure JSON serialization compatibility
        lora_params = convert_omegaconf_to_python(lora_params)

        # Additional manual conversion to ensure all types are correct
        lora_params = {
            "target_modules": list(lora_params["target_modules"]),
            "lora_alpha": int(lora_params["lora_alpha"]),
            "lora_dropout": float(lora_params["lora_dropout"]),
            "r": int(lora_params["r"]),
            "bias": str(lora_params["bias"]),
            "task_type": str(lora_params["task_type"]),
        }

        # Log converted parameters for debugging
        logger.info(f"Converted LoRA parameters: {lora_params}")
        logger.info(f"target_modules type: {type(lora_params['target_modules'])}")
        logger.info(f"target_modules content: {lora_params['target_modules']}")

        # Test JSON serialization to ensure compatibility
        try:
            import json

            json.dumps(lora_params)
            logger.info("✅ LoRA parameters are JSON serializable")
        except Exception as e:
            logger.error(f"❌ LoRA parameters are NOT JSON serializable: {e}")
            raise ValueError(f"LoRA parameters cannot be serialized to JSON: {e}")

        # Load the base model
        logger.info(f"Loading base model from {pretrained_model_name_or_path}")
        # Get HuggingFace token from .env or environment variables
        hf_token = get_hf_token()
        if hf_token and "token" not in kwargs:
            kwargs["token"] = hf_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Create LoRA configuration with converted parameters
        peft_config = LoraConfig(
            target_modules=lora_params["target_modules"],
            lora_alpha=lora_params["lora_alpha"],
            lora_dropout=lora_params["lora_dropout"],
            r=lora_params["r"],
            bias=lora_params["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        logger.info(f"Applying LoRA with config: {peft_config}")
        model = get_peft_model(base_model, peft_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        return model


def get_lora_model(model_cfg: DictConfig):
    """
    Load a model with LoRA adapters using the model configuration.

    Args:
        model_cfg: Model configuration containing model_args, tokenizer_args, and lora_config

    Returns:
        Tuple of (model, tokenizer)
    """
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )

    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    lora_config = model_cfg.get("lora_config", None)

    # Get torch dtype using the same logic as the main module
    torch_dtype = get_dtype(model_args)

    # Get HuggingFace token from .env or environment variables
    hf_token = get_hf_token()
    
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
        if hf_token and "token" not in model_args:
            model_args["token"] = hf_token

    try:
        # Load model with LoRA
        model = LoRAModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            lora_config=lora_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=hf_home,
            **model_args,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching LoRA model using LoRAModelForCausalLM.from_pretrained()."
        )

    # Load tokenizer using the same logic as the main module
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def get_dtype(model_args):
    """Extract torch dtype from model arguments."""
    with open_dict(model_args):
        torch_dtype_str = model_args.pop("torch_dtype", None)

    if torch_dtype_str is None:
        return torch.float32

    if torch_dtype_str == "bfloat16":
        return torch.bfloat16
    elif torch_dtype_str == "float16":
        return torch.float16
    elif torch_dtype_str == "float32":
        return torch.float32

    return torch.float32


def get_tokenizer(tokenizer_args):
    """Load tokenizer from tokenizer arguments."""
    # Get HuggingFace token from .env or environment variables
    hf_token = get_hf_token()
    tokenizer_kwargs = dict(tokenizer_args)
    if hf_token and "token" not in tokenizer_kwargs:
        tokenizer_kwargs["token"] = hf_token
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_args.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_args}\n"
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


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")
